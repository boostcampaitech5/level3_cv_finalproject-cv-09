This is README file to use FastAPI

Before read this instruction, you have to enter following command in terminal to import some packages :

$pip install uvicorn
$pip install fastapi

- To Execute

Run following command in terminal :
uvicorn main:app --reload

Now your brief server is on!

- APIs

There are three APIs.

1. upload
This API offers upload function.
If it runs properly, result will be following :
'File is saved in {Image_Dir}'

2. predict
This API offer predict function.
Predict is implemented by using CLIPSEG.
If is runs properly, a jpg file will be returned.
Also, the output will be saved in following path, using {image_id} :
'/opt/ml/level3_cv_finalproject-cv-09/FastAPI/predicts'

3. download
This API offer download function.
You can download an image by using image id, used in upload session.