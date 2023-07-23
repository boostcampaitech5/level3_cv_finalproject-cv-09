import os
import torch
import numpy as np
import shutil
import json
import cv2
import argparse
from torchvision import transforms
from io import BytesIO
from hrnet.models.light import PLModel
from hrnet.dataset import CustomKRLoadSegmentation
from torchvision.transforms import ToTensor, Normalize
from zipfile import ZipFile
from utils.tools_gradio import fast_process
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from PIL import Image
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from lang_segment_anything.lang_sam import LangSAM
from lang_segment_anything.lang_sam import SAM_MODELS
from lang_segment_anything.lang_sam.utils import draw_image
from lang_segment_anything.lang_sam.utils import load_image


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

    # Lang-SAM load
    app.state.lang_sam = LangSAM(sam_type="vit_h", device=device)


def change_path(path):
    if path.endswith(".png"):
        path = str(path.split(".")[0] + ".jpg")
    return path


def rle_encode(mask):
    """
    다차원 텐서를 RLE 인코딩하는 함수

    :param tensor: 2차원 텐서 (height x width)
    :return: RLE 인코딩된 문자열 리스트
    """
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
    parser = argparse.ArgumentParser(description="mlflow-pytorch test")
    parser.add_argument("--accelerator", choices=["cpu", "gpu", "auto"], default="gpu")
    parser.add_argument("--precision", choices=["32", "16"], default="16")
    parser.add_argument("--regist_name", type=str, default="register_model")
    parser.add_argument("--bn_type", type=str, default="torchbn")
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--backbone", type=str, default="hrnet48")
    parser.add_argument(
        "--pretrained",
        type=str,
        default="/opt/ml/level3_cv_finalproject-cv-09/MLflow/checkpoint/best.pth",
    )
    parser.add_argument("--experiment_name", type=str, default="mlflow_ex")

    args = parser.parse_args()
    return args


def process_image_and_get_masks(img):
    args = get_arg()

    # Load and preprocess the image
    convert_tensor = transforms.Compose(
        [ToTensor(), Normalize((0.286, 0.325, 0.283), (0.186, 0.190, 0.187))]
    )
    image = convert_tensor(img)
    image = image.unsqueeze(0)

    # Initialize the lighiting model
    model = PLModel(args=args)

    # Get masks using the 'test' function
    masks = test(model, image)
    return masks


def mask_color(mask, tuple):
    cmap = tuple.cmap
    if isinstance(mask,np.ndarray):
        r_mask = np.zeros_like(mask,dtype=np.uint8)
        g_mask = np.zeros_like(mask,dtype=np.uint8)
        b_mask = np.zeros_like(mask,dtype=np.uint8)
        for k in range(len(cmap)):
            indice = mask==k
            r_mask[indice] = cmap[k][0]
            g_mask[indice] = cmap[k][1]
            b_mask[indice] = cmap[k][2]
        return np.stack([b_mask, g_mask, r_mask], axis=2)


def hrnet_inference(id, file_name):
    img = Image.open(f"{FOLDER_DIR}/{id}/original/{file_name}")
    mask, mask_list = process_image_and_get_masks(img)
    rle_list = []
    for element in mask_list :
        temp = [element[0], rle_encode(np.array(element[1]))]
        rle_list.append(temp)
        
    # 이미지 저장
    out = np.squeeze(mask, axis=0)
    out = mask_color(out,CustomKRLoadSegmentation)
    output_path = f'{FOLDER_DIR}/{id}/hrnet/{file_name}'
    cv2.imwrite(output_path, out)

    return rle_list


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
    mask_dict = {"masks" : list(), "size": [new_h, new_w]}
    for idx, annotation in enumerate(annotations):
        rle_mask = rle_encode(annotation['segmentation'])
        mask_dict['masks'].append(rle_mask)
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


@torch.no_grad()
async def segment_dino(
    box_threshold=0.7, text_threshold=0.7, image_path="", text_prompt="sky"
):
    image_pil = load_image(image_path)  # width x height
    masks, boxes, phrases, logits = app.state.lang_sam.predict(
        image_pil, text_prompt, box_threshold, text_threshold
    )  # channel x height x width
    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
    mask_dict = {"masks" : dict(), "size": [image_pil.height, image_pil.width]}
    print(mask_dict['size'])

    for idx, label in enumerate(labels):
        label, logit = label.split()
        if label in mask_dict["masks"]:
            mask1 = np.array(mask_dict["masks"][label])
            mask2 = np.array(masks[idx])
            or_mask = np.logical_or(mask1, mask2)
            mask_dict["masks"][label] = torch.tensor(or_mask)
        else:
            mask_dict["masks"][label] = torch.tensor(masks[idx])
    for label, mask in mask_dict["masks"].items():
        rle_mask = rle_encode(mask)
        mask_dict["masks"][label] = rle_mask
    image_array = np.asarray(image_pil)
    image = draw_image(image_array, masks, boxes, labels)
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    return mask_dict, image


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
        if file.endswith(".png"):
            path = f"{FOLDER_DIR}/{id}/original/{file.split('.')[0]}"
            jpg_path = f"{path}.jpg"
            img = Image.open(f"{path}.png").convert("RGB")
            img.save(jpg_path)
            os.remove(f"{path}.png")


@app.post("/segment/")
async def segment(path: str = Form(...)):
    path = change_path(path)
    id, file_name = path.split("/")
    img_path = f"{FOLDER_DIR}/{id}/original/{file_name}"
    img = Image.open(img_path).convert("RGB")
    fig, mask_dict = await segment_everything(img)
    output = fig.convert("RGB")
    if not os.path.isdir(f"{FOLDER_DIR}/{id}/segment/"):
        os.mkdir(f"{FOLDER_DIR}/{id}/segment/")
    output.save(f"{FOLDER_DIR}/{id}/segment/{file_name}")
    # seg_img = FileResponse(
    #     f"{FOLDER_DIR}/{id}/segment/{file_name}",
    #     media_type="image/jpg",
    # )
    output_reponse = JSONResponse(content=mask_dict)
    return output_reponse


@app.post("/segment_text/")
async def segment_text(
    path: str = Form(...), text_prompt: str = Form(...), threshold: float = Form(...)
):
    path = change_path(path)
    id, file_name = path.split("/")
    img_path = f"{FOLDER_DIR}/{id}/original/{file_name}"
    text_prompt = text_prompt.replace(",", ".")
    text_seg_dict, segmented_image = await segment_dino(
        threshold, threshold, img_path, text_prompt=text_prompt
    )
    if not os.path.isdir(f"{FOLDER_DIR}/{id}/segment/"):
        os.mkdir(f"{FOLDER_DIR}/{id}/segment/")
    segmented_image.save(f"{FOLDER_DIR}/{id}/segment/dino_{file_name}")
    # mask_json = jsonable_encoder(text_seg_masks.tolist())
    # seg_dino_img = FileResponse(
    #     f"{FOLDER_DIR}/{id}/segment/dino_{file_name}",
    #     media_type="image/jpg",
    # )
    output_reponse = JSONResponse(content=text_seg_dict)
    return output_reponse


# Send data from FastAPI server to FE server
@app.post("/segment_hrnet/")
def segment_hrnet(path: str = Form(...)):
    path = change_path(path)
    id, file_name = path.split("/")
    
    rle_list = hrnet_inference(id, file_name)
    json_rle_list = json.dumps(rle_list, indent=2)
    # hrnet_img = FileResponse(
    #    f"{FOLDER_DIR}/{id}/hrnet/{file_name}",
    #    media_type="image/jpg",
    # )
    hrnet_json = JSONResponse(json_rle_list)
    # please check if multiple Response works
    # return hrnet_img, hrnet_json
    return hrnet_json


@app.post("/json_download/")
def json_download(path: str = Form(...)):
    id, file_name = path.split("/")
    path = change_path(path)
    file_name = file_name.split(".")[0]
    output = {"test": [1, 2, 3, 4], "test2": [5, 6, 7, 8]}
    with open(f"{FOLDER_DIR}/{id}/{file_name}.json", "w") as f:
        json.dump(output, f, indent=2)
    return output


@app.post("/remove/")
def remove(id: str = Form(...), annotated_data: dict = Form(...)):
    if id == "":
        return 0
    zip_file = ZipFile(f"{FOLDER_DIR}/{id}/{id}.zip", "w")
    for file in os.listdir(f"{FOLDER_DIR}/{id}/original"):
        zip_file.write(os.path.join(f"{FOLDER_DIR}/{id}/original", file))
    zip_file.close()
    """
    <TO BE IMPLEMENTED>
    Send zipfile to airflow server using scp command
    """
    path_list = []
    path_list.append(f"{FOLDER_DIR}/{id}/original")
    path_list.append(f"{FOLDER_DIR}/{id}/segment")
    path_list.append(f"{FOLDER_DIR}/{id}/zip")
    for path in path_list:
        if os.path.isdir(path):
            shutil.rmtree(path)
