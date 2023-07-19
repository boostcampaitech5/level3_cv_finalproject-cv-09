import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import shutil
import cv2
import json
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from utils.tools_gradio import fast_process
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from PIL import Image
from torchvision import transforms
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

    app.state.colors = [
        (0, 0, 0),
        (0.8196078431372549, 0.2901960784313726, 0.25882352941176473),
        (0.42745098039215684, 0.9490196078431372, 0.2),
        (0.9490196078431372, 0.9254901960784314, 0.8862745098039215),
        (0.5764705882352941, 0.19607843137254902, 0.6235294117647059),
        (0.0196078431372549, 0.41568627450980394, 0.9725490196078431),
        (0.3764705882352941, 0.20784313725490197, 0.09411764705882353),
        (0.12156862745098039, 0.4745098039215686, 0.38823529411764707),
        (0.00392156862745098, 0.34901960784313724, 0.01568627450980392),
        (0.4470588235294118, 0.00392156862745098, 0.03137254901960784),
        (0.32941176470588235, 0.34901960784313724, 0.7607843137254902),
    ]

    # Load the pre-trained model
    sam_checkpoint = "weights/mobile_sam.pt"
    model_type = "vit_t"

    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam = mobile_sam.to(device=device)
    mobile_sam.eval()

    app.state.mask_generator = SamAutomaticMaskGenerator(mobile_sam)
    app.state.predictor = SamPredictor(mobile_sam)

    # clip_seg load code
    # app.state.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    # app.state.model = CLIPSegForImageSegmentation.from_pretrained(
    #     "CIDAS/clipseg-rd64-refined"
    # )
    
    # Lang-SAM load
    app.state.lang_sam = LangSAM(sam_type="vit_h", device = device)
    


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
    image_pil = load_image(image_path)
    print("image size : ", image_pil.size) # width x height
    masks, boxes, phrases, logits = app.state.lang_sam.predict(image_pil, text_prompt, box_threshold, text_threshold)
    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
    print("masks : ", masks.shape)  # channel x height x width
    json_mask = json.dumps(masks.tolist())
    image_array = np.asarray(image_pil)
    image = draw_image(image_array, masks, boxes, labels)
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    return json_mask, image

# @torch.no_grad()
# def clip_segmentation(image, label_list):
#     inputs = app.state.processor(
#         text=label_list,
#         images=[image] * len(label_list),
#         padding="max_length",
#         return_tensors="pt",
#     )
#     outputs = app.state.model(**inputs)

#     preds = outputs.logits.unsqueeze(1).cpu()
#     flat_preds = torch.sigmoid(preds.squeeze()).reshape((preds.shape[0], -1))
#     flat_preds_with_treshold = torch.full(
#         (preds.shape[0] + 1, flat_preds.shape[-1]), 0.5
#     )  # threshold 변경 필요
#     flat_preds_with_treshold[1 : preds.shape[0] + 1, :] = flat_preds
#     inds = torch.topk(flat_preds_with_treshold, 1, dim=0).indices

#     temp_list = []
#     for i in inds.squeeze():
#         temp_list.append(app.state.colors[i])
#     image = cv2.resize(np.array(image), (preds.shape[-2], preds.shape[-1]))
#     output = (
#         np.array(temp_list)
#         .T.reshape(3, preds.shape[-2], preds.shape[-1])
#         .transpose(1, 2, 0)
#         * 255
#     )
#     blended = cv2.addWeighted(image, 0.5, output, 0.5, 0, dtype=cv2.CV_8UC3)
#     return np.clip(blended, 0, 255)


@app.post("/zip_upload/")
async def zip_upload(id: str = Form(...), files: UploadFile = File(...)):
    path_list = []
    path_list.append(f"{FOLDER_DIR}/{id}/original")
    path_list.append(f"{FOLDER_DIR}/{id}/segment")
    path_list.append(f"{FOLDER_DIR}/{id}/zip")

    for path in path_list:
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

    file_name = (files.filename).split(".")[0]
    content = await files.read()

    ZIP_PATH = f"{FOLDER_DIR}/{id}/zip"

    with open(f"{ZIP_PATH}/{file_name}.zip", "wb") as f:
        f.write(content)
    zipfile.ZipFile(f"{ZIP_PATH}/{file_name}.zip").extractall(f"data/{id}/original")


@app.post("/segment/")
async def segment(path: str = Form(...)):
    id, file_name = path.split("/")
    img_path = f"{FOLDER_DIR}/{id}/original/{file_name}"
    img = Image.open(img_path).convert("RGB")
    # if file_name.endswith(".png"):
    #     jpg_path = f"{file_name.split('.')[0]}.jpg"
    #     img.save(jpg_path)
    #     img = Image.open(jpg_path)
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
async def segment_text(path: str = Form(...), text_prompt: str = Form(...)):
    box_threshold, text_threshold = 0.3, 0.3
    id, file_name = path.split("/")
    img_path = f"{FOLDER_DIR}/{id}/original/{file_name}"
    if file_name.endswith(".png"):
        jpg_path = f"{file_name.split('.')[0]}.jpg"
        img = Image.open(img_path).convert("RGB")
        img.save(jpg_path)
        img_path = jpg_path
    text_seg_masks, segmented_image = await segment_dino(box_threshold, text_threshold, img_path, text_prompt = text_prompt)
    if not os.path.isdir(f"{FOLDER_DIR}/{id}/segment/"):
        os.mkdir(f"{FOLDER_DIR}/{id}/segment/")
    segmented_image.save(f"{FOLDER_DIR}/{id}/segment/dino_{file_name}")
    # mask_json = jsonable_encoder(text_seg_masks.tolist())
    # seg_dino_img = FileResponse(
    #     f"{FOLDER_DIR}/{id}/segment/dino_{file_name}",
    #     media_type="image/jpg",
    # )
    output_reponse = JSONResponse(content=text_seg_masks)
    return output_reponse
    

@app.post("/json_download/")
def json_download(path: str = Form(...)):
    id, file_name = path.split("/")
    file_name = file_name.split(".")[0]
    output = {
        "test" : [1, 2, 3, 4],
        "test2" : [5, 6, 7, 8]
    }
    with open(f'{FOLDER_DIR}/{id}/{file_name}_segment.json' ,'w') as f:
        json.dump(output, f, indent=2)
    return output


@app.post("/remove/")
def remove(id: str = Form(...)):
    if id == "":
        return 0
    path_list = []
    path_list.append(f"{FOLDER_DIR}/{id}/original")
    path_list.append(f"{FOLDER_DIR}/{id}/segment")
    path_list.append(f"{FOLDER_DIR}/{id}/zip")
    for path in path_list:
        if os.path.isdir(path):
            shutil.rmtree(path)

