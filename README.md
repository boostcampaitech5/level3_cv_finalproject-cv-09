# Data Annotation Tool Continous Development

>   부스트캠프 5기 CV-09 최종 프로젝트
>   개발기간 : 2023.06.28 - 2023.07.??

---
### 프로젝트 소개

    HRNET과 Mobile SAM을 이용하여 쉽고 편리한 라벨링을 수행하는 Annotation Tool입니다.

##### 제작 동기
    양질의 데이터에 대한 필요성과 중요도가 증가하고 있으나 비전문가가 편리하게 이미지에 대한 라벨링을 수행하고 도구가 부족합니다.
    데이터 제작부터 재훈련까지의 MLOps 사이클을 경험하고자 합니다.

##### 목표
    Segmentation Annotation 하는 과정에서 학습된 모델로 Pseudo Labelling을 수행하고 해당 모델을 지속적으로 발전시켜 양질의 데이터 제작 서비스 제공을 목표로 합니다.


---
### 시작 가이드
본 프로젝트는 Gradio, FastAPI, MLFlow, AirFlow로 구성되어 있습니다.
##### Gradio
```
cd gradio
pip install -r requirements.txt
```
##### FastAPI
##### Install Mobile SAM(for Backend Server)

```
git clone git@github.com:ChaoningZhang/MobileSAM.git
cd MobileSAM; pip install -e .
```
```
cd ..
cd FastAPI
pip install -r requirements.txt
```

##### MLFlow
```
cd MLflow
pip install -r requirements.txt
```

##### AirFlow
```
cd Airflow
pip install -r requirements.txt
```

---
### 기술 스택

##### Environment
<img src="https://img.shields.io/badge/visual studio code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white">
<img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">

##### Communication
<img src="https://img.shields.io/badge/notion-000000?style=for-the-badge&logo=notion&logoColor=white">
<img src="https://img.shields.io/badge/slack-4A154B?style=for-the-badge&logo=slack&logoColor=white">

##### Development
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
<img src="https://img.shields.io/badge/mlflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white">
<img src="https://img.shields.io/badge/airflow-017CEE?style=for-the-badge&logo=apacheairflow&logoColor=white">

---
### 화면 구성

![image1](gradio/assets/image1.png)
![image2](gradio/assets/image2.png)

---
### 주요 기능 설명

##### 1. Pseudo Labelling
>    HRNet을 이용한 Pseudo Labelling 기능을 제공합니다.
>    Relabelling 작업의 가이드 역할을 합니다.

##### 2. Relabelling
>    Mobile SAM을 이용한 Relabelling 기능을 제공합니다.
>    사용자가 직접 라벨을 수정할 수 있습니다.

##### 3. Retrain
>    MLFlow 및 AirFlow를 이용하여 Retrain을 수행합니다.


### 디렉토리 구조
```
.
|-- Airflow
|   |-- dags
|   `-- func
|-- FastAPI
|   |-- __pycache__
|   |-- data
|   |-- hrnet
|   |-- lang_segment_anything
|   |-- mobile_sam
|   |-- utils
|   `-- weights
|-- MLflow
|   |-- checkpoint
|   |-- dataset
|   `-- models
|-- MobileSAM
|   |-- app
|   `-- mobile_sam
`-- gradio
    |-- __pycache__
    |-- assets
    `-- data
```
