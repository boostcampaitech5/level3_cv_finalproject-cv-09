import os
import torch
import numpy as np
import shutil
import json
from PIL import Image
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
    app.state.lang_sam = LangSAM(sam_type="vit_h", device = device)

def change_path(path):
    if path.endswith('.png'):
        path = str(path.split('.')[0] + '.jpg')
    return path
        

def rle_encode(mask):
    """
    다차원 텐서를 RLE 인코딩하는 함수

    :param tensor: 3차원 텐서 (channel x height x width)
    :return: RLE 인코딩된 문자열 리스트
    """
    mask_flatten = mask.flatten()
    mask_flatten = np.concatenate([[0], mask_flatten, [0]])
    runs = np.where(mask_flatten[1:] != mask_flatten[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join(str(x) for x in runs)
    return rle


@torch.no_grad()
async def segment_everything(
    image,
    input_size=1024,
    better_quality=True,
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
    return fig


@torch.no_grad()
async def segment_dino(box_threshold = 0.7, text_threshold = 0.7, image_path = "", text_prompt = "sky"):
    print("text : ", text_prompt)
    image_pil = load_image(image_path)  # width x height
    masks, boxes, phrases, logits = app.state.lang_sam.predict(image_pil, text_prompt, box_threshold, text_threshold)   # channel x height x width
    for idx, phrase in enumerate(phrases):
        phrases[idx] = phrase.replace(" ", "_")
    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
    mask_dict = dict()
    for idx, label in enumerate(labels):
        label, logit = label.split()
        print(label, logit)
        print(masks[idx].shape)
        if label in mask_dict:
            mask1 = np.array(mask_dict[label])
            mask2 = np.array(masks[idx].tolist())
            or_mask = np.logical_or(mask1, mask2)
            mask_dict[label] = or_mask.tolist()
        else:
            mask_dict[label] = masks[idx].tolist()
    rle_mask = rle_encode(masks)
    json_rle = json.dumps(rle_mask)
    json_mask = json.dumps(masks.tolist())
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
    # path_list.append(f"{FOLDER_DIR}/{id}/hrnet")

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
        if file.endswith('.png'):
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
    output = await segment_everything(img)
    output = output.convert("RGB")
    if not os.path.isdir(f"{FOLDER_DIR}/{id}/segment/"):
        os.mkdir(f"{FOLDER_DIR}/{id}/segment/")
    output.save(f"{FOLDER_DIR}/{id}/segment/{file_name}")
    seg_img = FileResponse(
        f"{FOLDER_DIR}/{id}/segment/{file_name}",
        media_type="image/jpg",
    )
    return seg_img


@app.post("/segment_text/")
async def segment_text(path: str = Form(...), text_prompt: str = Form(...), threshold: float = Form(...)):
    path = change_path(path)
    id, file_name = path.split("/")
    img_path = f"{FOLDER_DIR}/{id}/original/{file_name}"
    text_prompt = text_prompt.replace(",", ".")
    text_seg_dict, segmented_image = await segment_dino(threshold, threshold, img_path, text_prompt = text_prompt)
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

# FE -> FastAPI : zip_upload
# FastAPI -> MLflow : scp? api?
# ML flow : inference
# ML flow -> FastAPI : hrnet
# FastAPI -> FE : segment_hrnet

# Send data from FastAPI server to FE server
# @app.post("/segment_hrnet/")
# def segment_hrnet(path: str = Form(...)):
#     path = change_path(path)
#     id, file_name = path.split("/")
#     pass

# Send data from MLFlow server to FastAPI server
@app.post("/hrnet/")
def hrnet(mask: str = Form(...), files: UploadFile = File(...)):
    path = f"{FOLDER_DIR}/hrnet"
    content = files.read()
    with open(path, "wb") as f:
        f.write(content)
    print(mask)

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
def remove(id: str = Form(...)):
    if id == "":
        return 0
    zip_file = ZipFile(f"{FOLDER_DIR}/{id}/{id}.zip", 'w')
    for file in os.listdir(f"{FOLDER_DIR}/{id}/original"):
        zip_file.write(os.path.join(f"{FOLDER_DIR}/{id}/original", file))
    zip_file.close()
    '''
    <TO BE IMPLEMENTED>
    Send zipfile to airflow server using scp command
    '''
    path_list = []
    path_list.append(f"{FOLDER_DIR}/{id}/original")
    path_list.append(f"{FOLDER_DIR}/{id}/segment")
    path_list.append(f"{FOLDER_DIR}/{id}/zip")
    for path in path_list:
        if os.path.isdir(path):
            shutil.rmtree(path)
