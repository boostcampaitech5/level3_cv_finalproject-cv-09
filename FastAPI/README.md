This is README file to use FastAPI

Before read this instruction, you have to enter following command in terminal to import some packages :

$pip install uvicorn
$pip install fastapi

- To Execute

Run following command in terminal :
uvicorn main:app --port {port_number} --reload

Now your brief server is on!

- APIs

There are four APIs.

1. upload
This API offers upload function.
If it runs properly, result will be following :
'File is saved in {Image_Dir}'

2. predict
This API offer predict function.
Predict is implemented by using CLIPSEG.
If is runs properly, a jpg file will be returned.
Also, the output will be saved in following path, using {image_id} :
'/opt/ml/level3_cv_finalproject-cv-09/FastAPI/data/{image_id}/predict.jpg'

Predict API will be removed soon since it is implemented not by communicating with client, but by using server.

* Error
- Output image is very noisy

3. download
This API offer download function.
You can download an image by using image id, used in upload session.

4. feedback
This API offer feedback function.
It is possible to send review or feedback about the annotation result using the API.

- Data
All of data is saved in following path according to {image_id}:
'/opt/ml/level3_cv_finalproject-cv-09/FastAPI/data/{image_id}'

Lists of data is following :
image.jpg : An image user upload.
predict.jpg : An image implementing predict API.
log.txt : Logging user's feedback by using feedback API.