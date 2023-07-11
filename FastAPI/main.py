import os
import torch
import zipfile
import shutil
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms
from models.clip_seg import CLIPDensePredT
from pydantic import BaseModel

FOLDER_DIR = '/opt/ml/level3_cv_finalproject-cv-09/FastAPI/data/'

app = FastAPI()

"""
멘토님의 작품
"""
# @app.on_event('start_up')
# def startup_event():
#     app.state.model = Asdf
    
def prediction(path, prompts, model):
    prompt = list(prompts.split(','))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])
    image = Image.open(path)
    img = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        repeat_num = len(prompt)
        preds = model(img.repeat(repeat_num,1,1,1).cuda(), prompt)[0]

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
@app.post('/upload/{system}')
async def upload(system: str, files: UploadFile = File(...)):
    content = await files.read()
    FOLDER = FOLDER_DIR + system
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)
    with open(f'{FOLDER}/test.zip', 'wb') as f:
        f.write(content)
    f.close()
    with zipfile.ZipFile(f'/opt/ml/level3_cv_finalproject-cv-09/FastAPI/data/{system}/test.zip', "r") as zip_ref:
        if system == 'mac':
            zip_ref.extractall(f"data/{system}")
        else :
            zip_ref.extractall(f"data/{system}/test")
    if system == 'mac':
        dummy = FOLDER + '/__MACOSX'
        if os.path.isdir(dummy):
            shutil.rmtree(dummy)

'''
Implement Model
'''
@app.post('/predict/{image_id}/{prompts}')
def predict(image_id: str, prompts: str):
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True).cpu()
    model.eval()

    model.load_state_dict(torch.load('weights/rd64-uni-refined.pth', map_location=torch.device('cpu')), strict=False);
    
    FOLDER = FOLDER_DIR + f'{image_id}'
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)
    path = os.path.join(FOLDER, 'image.jpg')
    preds = prediction(path, prompts, model).cpu()
    output = visualize_segmentation(preds)
    print(output)
    pil_t = transforms.ToPILImage()
    output_pil = pil_t(output)
    print(output_pil.type)
    file_name = 'predict.jpg'
    path = os.path.join(FOLDER, file_name)
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(output)
    plt.savefig(path)
    
    return FileResponse(path)

# create table user_requests 
# (
#     task_id sequence 1,2,3,4
#     user_id  ~
#     img_file_path # s3, minio같은 경로
#     original_annotation json not null
#     updated_annotation json nullable
#     created_at
#     updated_at
    
# )
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
