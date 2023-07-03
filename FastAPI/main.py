from typing import Union
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

IMG_DIR = '/opt/Annotation_Demo/images/N-B-P-021_000109.jpg'
FOLDER_DIR = '/opt/Annotation_Demo/FastAPI/images'
app = FastAPI()

# Upload an image
@app.post('/upload/{image_id}')
async def upload(file: UploadFile, image_id: int):
    filename = file.filename
    content = await file.read()
    with open(os.path.join(FOLDER_DIR, filename), 'wb') as f:
        f.write(content)
    result = 'File is saved in ' + FOLDER_DIR + filename
    return result

# Execute the model
@app.post('/predict/')
def predict():
    pass

#Download the result
@app.get('/download/{name}')
def predict(name: str):
    path = os.path.join(FOLDER_DIR, str(name) + '.png')
    return FileResponse(path)