import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import shutil
import datetime
import cv2
from pytz import timezone
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from utils.tools_gradio import fast_process
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms
from models.clip_seg import CLIPDensePredT
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

FOLDER_DIR = '/opt/ml/level3_cv_finalproject-cv-09/FastAPI/data'

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
    app.state.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    app.state.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    
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
        use_retina=use_retina
    )
    return fig

@torch.no_grad()
def clip_segmentation(image, label_list):
    inputs = app.state.processor(
        text=label_list,
        images=[image] * len(label_list),
        padding="max_length",
        return_tensors="pt",
    )
    outputs = app.state.model(**inputs)

    preds = outputs.logits.unsqueeze(1).cpu()
    flat_preds = torch.sigmoid(preds.squeeze()).reshape((preds.shape[0], -1))
    flat_preds_with_treshold = torch.full(
        (preds.shape[0] + 1, flat_preds.shape[-1]), 0.5
    )  # threshold 변경 필요
    flat_preds_with_treshold[1 : preds.shape[0] + 1, :] = flat_preds
    inds = torch.topk(flat_preds_with_treshold, 1, dim=0).indices

    temp_list = []
    for i in inds.squeeze():
        temp_list.append(app.state.colors[i])
    image = cv2.resize(np.array(image), (preds.shape[-2], preds.shape[-1]))
    output = (
        np.array(temp_list)
        .T.reshape(3, preds.shape[-2], preds.shape[-1])
        .transpose(1, 2, 0)
        * 255
    )
    blended = cv2.addWeighted(image, 0.5, output, 0.5, 0, dtype=cv2.CV_8UC3)
    return np.clip(blended, 0, 255)

class Log(BaseModel):
    log: str

class Segment(BaseModel):
    input_size_slider: int

class User(BaseModel):
    id: str

@app.post('/zip_upload/')
async def zip_upload(id: int, files: UploadFile = File(...)):
    content = await files.read()
    
    # Try to make a directory
    if not os.path.isdir(FOLDER_DIR):
        os.mkdir(FOLDER_DIR)
    
    # Write the files in the file
    # id = item.id
    file_name = str(datetime.datetime.now(timezone('Asia/Seoul'))) + str(id)
    print(file_name)
    with open(f'{FOLDER_DIR}/zip/{file_name}.zip', 'wb') as f:
        f.write(content)
    f.close()
    zipfile.ZipFile(f'{FOLDER_DIR}/zip/{file_name}.zip').extractall('data/original/{file_name}')
    
    # To erase MAC dummy file
    # dummy = FOLDER_DIR + '/__MACOSX'
    # if os.path.isdir(dummy):
    #     shutil.rmtree(dummy)

# Implement Model

@app.get('/segment/')
async def segment():
    zip_list = os.listdir(f'{FOLDER_DIR}/zip')
    original_list = os.listdir(f'{FOLDER_DIR}/original')
    
    img = Image.open(f'{FOLDER_DIR}/test/1.jpg')
    output = await segment_everything(img, 1024)
    output = output.convert("RGB")
    output.save("data/segment/predict.jpg")
    return FileResponse(f'{FOLDER_DIR}/segment/prdict.jpg')

# @app.post('/predict/')
# def predict(image_id: str, prompts: str):
#     model = CLIPDensePredT(version='ViT-B/32', reduce_dim=64).cuda()
#     model.eval()

#     model.load_state_dict(torch.load('weights/rd64-uni-refined.pth', map_location=torch.device('cpu')), strict=False);
    
#     FOLDER = FOLDER_DIR + f'{image_id}'
#     if not os.path.isdir(FOLDER):
#         os.mkdir(FOLDER)
#     path = os.path.join(FOLDER, 'image.jpg')
#     preds = prediction(path, prompts, model).cpu()
#     output = visualize_segmentation(preds)
#     file_name = 'predict.jpg'
#     path = os.path.join(FOLDER, file_name)
#     plt.figure(figsize=(10, 10))
#     plt.axis('off')
#     plt.imshow(output)
#     plt.savefig(path)
    
#     return FileResponse(path)
    
# Download the result

# @app.get('/download/{image_id}')
# def download(image_id: str):
#     FOLDER = FOLDER_DIR + f'{image_id}'
#     file_name = 'predict.jpg'
#     path = os.path.join(FOLDER, file_name)
#     return FileResponse(path)

    
# Send Feedback

# @app.post('/log/{image_id}')
# def log(image_id: str, log: Log):
#     FOLDER = FOLDER_DIR + f'{image_id}'
#     file_name = 'log.txt'
#     path = os.path.join(FOLDER, file_name)
#     with open(path, 'w') as f:
#         f.write(log.log)