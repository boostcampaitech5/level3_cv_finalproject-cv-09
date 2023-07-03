from typing import Union
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch
import cv2
import numpy as np
from torchvision import transforms
from models.clip_seg import CLIPDensePredT
import matplotlib.pyplot as plt
import io

IMG_DIR = '/opt/level3_cv_finalproject-cv-09/images/N-B-P-021_000109.jpg'
FOLDER_DIR = '/opt/ml/level3_cv_finalproject-cv-09/FastAPI/images'
PREDICT_DIR = '/opt/ml/level3_cv_finalproject-cv-09/FastAPI/predicts'
app = FastAPI()

def prediction(path, prompts, model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])
    image = Image.open(path)
    img = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        repeat_num = len(prompts)
        preds = model(img.repeat(repeat_num,1,1,1).cuda(), prompts)[0]
        
    # print("pred shape : ", preds.shape)
    return preds

def visualize_segmentation(preds, threshold = 0.5):
    flat_preds = torch.sigmoid(preds.squeeze()).reshape((preds.shape[0], -1))

    # Initialize a dummy "unlabeled" mask with the threshold
    flat_preds_with_treshold = torch.full((preds.shape[0] + 1, flat_preds.shape[-1]), threshold)
    flat_preds_with_treshold[1:preds.shape[0]+1,:] = flat_preds

    # Get the top mask index for each pixel
    inds = torch.topk(flat_preds_with_treshold, 1, dim=0).indices.reshape((preds.shape[-2], preds.shape[-1]))

    # segmentation_figure = plt.figure(figsize=(10, 10))
    # plt.axis('off')
    # plt.imshow(inds)
    # img_buf = io.BytesIO()
    # plt.savefig('predict.jpg')
    # plt.close(segmentation_figure)
    # return img_buf
    # st.pyplot(segmentation_figure)
    return inds

# Upload an image
@app.post('/upload/{image_id}')
async def upload(file: UploadFile, image_id: int):
    filename = file.filename
    content = await file.read()
    image_id = str(image_id) + '.jpg'
    with open(os.path.join(FOLDER_DIR, image_id), 'wb') as f:
        f.write(content)
    result = 'File is saved in ' + FOLDER_DIR + image_id
    return result

# Execute the model
@app.post('/predict/{image_id}')
def predict(image_id: str, prompts: str):
    model = CLIPDensePredT(version='ViT-B/32', reduce_dim=64).cuda()
    model.eval();

    model.load_state_dict(torch.load('weights/rd64-uni-refined.pth', map_location=torch.device('cpu')), strict=False);
    
    path = os.path.join(FOLDER_DIR, image_id + '.jpg')
    preds = prediction(path, prompts, model).cpu()
    output = visualize_segmentation(preds)
    
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(output)
    plt.savefig(f'predicts/{image_id}.jpg')
    
    return FileResponse(f'predicts/{image_id}.jpg')
    
#Download the result
@app.get('/download/{image_id}')
def predict(image_id: str):
    path = os.path.join(PREDICT_DIR, image_id + '.jpg')
    return FileResponse(path)