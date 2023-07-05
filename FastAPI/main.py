from typing import Union
import os
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import torch
from torchvision import transforms
from models.clip_seg import CLIPDensePredT
import matplotlib.pyplot as plt
from pydantic import BaseModel

IMG_DIR = '/opt/level3_cv_finalproject-cv-09/images/N-B-P-021_000109.jpg'
FOLDER_DIR = '/opt/ml/level3_cv_finalproject-cv-09/FastAPI/data/'

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

    return preds

def visualize_segmentation(preds, threshold = 0.5):
    flat_preds = torch.sigmoid(preds.squeeze()).reshape((preds.shape[0], -1))

    # Initialize a dummy "unlabeled" mask with the threshold
    flat_preds_with_treshold = torch.full((preds.shape[0] + 1, flat_preds.shape[-1]), threshold)
    flat_preds_with_treshold[1:preds.shape[0]+1,:] = flat_preds

    # Get the top mask index for each pixel
    inds = torch.topk(flat_preds_with_treshold, 1, dim=0).indices.reshape((preds.shape[-2], preds.shape[-1]))

    return inds

class Log(BaseModel):
    log: str

'''    
Upload an image
'''
@app.post('/upload/{image_id}')
async def upload(file: UploadFile, image_id: str):
    content = await file.read()
    
    FOLDER = FOLDER_DIR + f'{image_id}'
    file_name = 'image.jpg'
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)
    path = os.path.join(FOLDER, file_name)
    with open(path, 'wb') as f:
        f.write(content)
    f.close()
    result = 'File is saved in ' + FOLDER_DIR + image_id
    return result

'''
Implement Model
'''
@app.post('/predict/{image_id}_{prompts}')
def predict(image_id: str, prompts: str):
    model = CLIPDensePredT(version='ViT-B/32', reduce_dim=64).cuda()
    model.eval()

    model.load_state_dict(torch.load('weights/rd64-uni-refined.pth', map_location=torch.device('cpu')), strict=False);
    
    FOLDER = FOLDER_DIR + f'{image_id}'
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)
    path = os.path.join(FOLDER, 'image.jpg')
    preds = prediction(path, prompts, model).cpu()
    output = visualize_segmentation(preds)
    file_name = 'predict.jpg'
    path = os.path.join(FOLDER, file_name)
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(output)
    plt.savefig(path)
    
    return FileResponse(path)
'''    
Download the result
'''
@app.get('/download/{image_id}')
def download(image_id: str):
    FOLDER = FOLDER_DIR + f'{image_id}'
    file_name = 'predict.jpg'
    path = os.path.join(FOLDER, file_name)
    return FileResponse(path)

'''    
Send Feedback
'''
@app.post('/log/{image_id}')
def log(image_id: str, log: Log):
    FOLDER = FOLDER_DIR + f'{image_id}'
    file_name = 'log.txt'
    path = os.path.join(FOLDER, file_name)
    with open(path, 'w') as f:
        f.write(log.log)