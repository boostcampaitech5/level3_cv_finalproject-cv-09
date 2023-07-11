import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import shutil
from utils.tools_gradio import fast_process
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms
from models.clip_seg import CLIPDensePredT
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

FOLDER_DIR = '/opt/ml/level3_cv_finalproject-cv-09/FastAPI/data/'

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

    mask_generator = SamAutomaticMaskGenerator(mobile_sam)
    predictor = SamPredictor(mobile_sam)
    
@torch.no_grad()
async def segment_everything(
    image,
    input_size=1024,
    better_quality=True,
    use_retina=True,
    mask_random_color=True,
):
    global mask_generator

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    nd_image = np.array(image)
    annotations = mask_generator.generate(nd_image)

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
class Log(BaseModel):
    log: str

@app.post('/zip_upload/')
async def zip_upload(files: UploadFile = File(...)):
    content = await files.read()
    
    # Try to make a directory
    if not os.path.isdir(FOLDER_DIR):
        os.mkdir(FOLDER_DIR)
        
    # Write the files in the file
    with open(f'{FOLDER_DIR}/test.zip', 'wb') as f:
        f.write(content)
    f.close()
    zipfile.ZipFile(f'{FOLDER_DIR}test.zip').extractall('data/test')

    # dummy = FOLDER_DIR + '/__MACOSX'
    # if os.path.isdir(dummy):
    #     shutil.rmtree(dummy)

# Implement Model

# @app.post('/segment/')
# def segment(cond_img_e: UploadFile,
#             input_size_slider: int,
#             mor_check: bool,
#             ratina_check: bool):
#     return segment_everything(cond_img_e, input_size_slider, mor_check, ratina_check)

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