import numpy as np
import os
import shutil
import torch
from collections import namedtuple
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from hrnet.models.light import PLModel
from hrnet.dataset import CustomKRLoadSegmentation
from PIL import Image
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from torchvision.transforms import ToTensor, Normalize, Compose
from utils.tools_gradio import fast_process
from utils.tools import box_prompt, format_results, point_prompt
from zipfile import ZipFile


FOLDER_DIR = "data"

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("startup")
async def startup_event():
    # Load the pre-trained model
    sam_checkpoint = "weights/mobile_sam.pt"
    model_type = "vit_t"

    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam = mobile_sam.to(device=device)
    mobile_sam.eval()

    app.state.mask_generator = SamAutomaticMaskGenerator(mobile_sam)
    app.state.predictor = SamPredictor(mobile_sam)


def change_path(path):
    if path.endswith(".jpg"):
        path = str(path.split(".")[0] + ".png")
    return path


def rle_encode(mask):
    mask_flatten = mask.flatten()
    mask_flatten = np.concatenate([[0], mask_flatten, [0]])
    runs = np.where(mask_flatten[1:] != mask_flatten[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = " ".join(str(x) for x in runs)
    return rle


def test(model, image):
    args = get_arg()
    model = model.cuda()
    model.eval()
    mask_list = []

    with torch.no_grad():
        n_class = args.num_classes
        image = image.cuda()
        logits = model(image)

        # restore original size
        outputs = torch.sigmoid(logits)
        outputs = outputs.argmax(dim=1)
        outputs = outputs.detach().cpu().numpy()

        result = outputs == np.arange(logits.shape[1])[:, np.newaxis, np.newaxis]
        for i in range(n_class):
            mask_list.append([CustomKRLoadSegmentation.label[i], result[i]])

    return outputs, mask_list


def get_arg():
    parser_dict = {
        "accelerator": "gpu",
        "precision": "16",
        "regist_name": "register_model",
        "bn_type": "torchbn",
        "num_classes": 19,
        "backbone": "hrnet48",
        "pretrained": "weights/best.pth",
    }
    ParserDict = namedtuple("ParserDict", parser_dict.keys())

    # 기존의 dictionary를 named tuple로 변환합니다.
    parser_dict = ParserDict(**parser_dict)

    return parser_dict


def process_image_and_get_masks(img):
    args = get_arg()

    # Load and preprocess the image
    convert_tensor = Compose(
        [ToTensor(), Normalize((0.286, 0.325, 0.283), (0.186, 0.190, 0.187))]
    )
    image = convert_tensor(img)
    image = image.unsqueeze(0)

    # Initialize the lighiting model
    model = PLModel(args=args)

    # Get masks using the 'test' function
    masks = test(model, image)
    return masks


def hrnet_inference(id, file_name):
    img = Image.open(f"{FOLDER_DIR}/{id}/original/{file_name}")
    mask, mask_list = process_image_and_get_masks(img)
    mask_dict = {"masks": dict(), "size": [img.height, img.width]}

    for element in mask_list:
        mask_dict["masks"][element[0]] = rle_encode(np.array(element[1]))

    return mask_dict


@torch.no_grad()
def segment_with_points(image, global_points, global_point_label, input_size=1024):
    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    scaled_points = np.array(
        [int(int(point) * scale) for point in global_points]
    ).reshape(-1, 2)
    scaled_point_label = np.array(
        [True if x == "True" else False for x in global_point_label]
    )

    if scaled_points.size == 0 and scaled_point_label.size == 0:
        print("No points selected")
        return image, image

    nd_image = np.array(image)
    app.state.predictor.set_image(nd_image)
    masks, scores, logits = app.state.predictor.predict(
        point_coords=scaled_points,
        point_labels=scaled_point_label,
        multimask_output=True,
    )

    results = format_results(masks, scores, logits, 0)

    annotations, _ = point_prompt(
        results, scaled_points, scaled_point_label, new_h, new_w
    )
    annotations = np.array(annotations)
    return annotations


@torch.no_grad()
async def segment_everything(
    image,
    input_size=1024,
    better_quality=False,
    use_retina=True,
    mask_random_color=True,
):
    # global mask_generator

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    nd_image = np.array(image)
    annotations = app.state.mask_generator.generate(nd_image)
    mask_dict = {"masks": list(), "size": [new_h, new_w]}
    for idx, annotation in enumerate(annotations):
        rle_mask = rle_encode(annotation["segmentation"])
        mask_dict["masks"].append(rle_mask)
    fig = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
    )
    return fig, mask_dict


@app.post("/zip_upload/")
async def zip_upload(id: str = Form(...), files: UploadFile = File(...)):
    path_list = []
    path_list.append(f"{FOLDER_DIR}/{id}/original")
    path_list.append(f"{FOLDER_DIR}/{id}/segment")
    path_list.append(f"{FOLDER_DIR}/{id}/zip")
    path_list.append(f"{FOLDER_DIR}/{id}/hrnet")

    for path in path_list:
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

    file_name = (files.filename).split(".")[0]
    content = await files.read()

    ZIP_PATH = f"{FOLDER_DIR}/{id}/zip"

    with open(f"{ZIP_PATH}/{file_name}.zip", "wb") as f:
        f.write(content)
    ZipFile(f"{ZIP_PATH}/{file_name}.zip").extractall(f"data/{id}/original")
    # Convert PNG to JPG
    for file in os.listdir(f"{FOLDER_DIR}/{id}/original/"):
        if file.endswith(".jpg"):
            path = f"{FOLDER_DIR}/{id}/original/{file.split('.')[0]}"
            png_path = f"{path}.png"
            img = Image.open(f"{path}.jpg").convert("RGB")
            img.save(png_path)
            os.remove(f"{path}.jpg")


@app.post("/segment/")
async def segment(
    path: str = Form(...),
    global_points: list = Form(...),
    global_point_label: list = Form(...),
):
    path = change_path(path)
    id, file_name = path.split("/")
    img_path = f"{FOLDER_DIR}/{id}/original/{file_name}"
    img = Image.open(img_path).convert("RGB")
    mask_array = segment_with_points(
        img, global_points=global_points, global_point_label=global_point_label
    )
    output_reponse = JSONResponse(content=mask_array.tolist())
    return output_reponse


# Send data to FE server
@app.post("/segment_hrnet/")
def segment_hrnet(path: str = Form(...)):
    path = change_path(path)
    id, file_name = path.split("/")
    rle_dict = hrnet_inference(id, file_name)
    hrnet_json = JSONResponse(content=rle_dict)
    return hrnet_json


# Upload data from FE server
@app.post("/json_upload/")
async def json_upload(id: str = Form(...), files: UploadFile = File(...)):
    file_name = (files.filename).split(".")[0]
    content = await files.read()
    ZIP_PATH = f"{FOLDER_DIR}/{id}/zip"

    with open(f"{ZIP_PATH}/{file_name}.zip", "wb") as f:
        f.write(content)
    ZipFile(f"{ZIP_PATH}/{file_name}.zip").extractall(f"data/{id}/original/annotations")


@app.post("/remove/")
def remove(id: str = Form(...)):
    if id == "":
        return 0
    os.rename(f"data/{id}/original", f"data/{id}/{id}")
    shutil.make_archive(f"data/{id}/{id}", "zip", f"data/{id}", f"{id}")

    # Implement SCP
    os.system(
        f"scp -P 2251 -i ../scp_key {FOLDER_DIR}/{id}/{id}.zip root@118.67.132.218:/opt/ml/level3_cv_finalproject-cv-09/MLflow/data/new_data"
    )

    path = f"{FOLDER_DIR}/{id}"
    if os.path.isdir(path):
        shutil.rmtree(path)


@app.post("/weight/")
async def weight(files: UploadFile = File(...)):
    file_name = files.filename
    content = await files.read()
    with open(
        f"/opt/ml/level3_cv_finalproject-cv-09/FastAPI/hrnet/checkpoint/{file_name}",
        "wb",
    ) as f:
        f.write(content)
