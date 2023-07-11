# README.md

부스트캠프 5기 CV-09 최종 프로젝트 FastAPI README 입니다.

README를 읽기 전에, 터미널에 다음 명령어를 입력하여 필요한 패키지를 설치하세요 :

    $pip install uvicorn
    $pip install fastapi
    $pip install pydantic
    $pip install torchvision

# 실행

    터미널에 다음 명령어를 입력하여 실행하세요 :
    uvicorn main:app --port {port_number} --reload
    
    외부 접속을 허용하기 위해서는 다음 명령어를 입력하세요 :
    uvicorn main:app --host 0.0.0.0 --port {port_number} --reload
    서버의 IP 주소를 이용하여 외부에서 접근이 가능합니다.
    {IP_Address}/{port_number}/docs로 이동하여 Swagger Docs 확인이 가능합니다.

# APIs

    현재 4개의 API를 제공하고 있습니다.
    upload API는 정상적으로 동작합니다.
    predict, download, feedback API는 정상적으로 동작하지 않습니다.

##### 구현 예정 기능
    - 로그인과 유사한 API를 제공하여 특정 유저가 다른 유저의 데이터에 접근할 수 없도록 합니다.

### 1. zip_upload

    zip 파일 업로드 기능을 제공하는 API입니다.
    zip 파일을 입력값으로 받아 지정된 경로에 zip 파일을 저장하고 파일의 압축을 해제합니다.

##### 수정 예정
    - 유저에 따라 파일 업로드 경로를 차별화합니다.
---
### 2. predict

    예측 기능을 제공하는 API입니다.
    사용자로부터 image_id, prompts를 입력값으로 받아 CLIPSEG를 이용하여 객체를 Segment하고, 결과를 저장합니다.

##### 에러
    - 출력이 정상적이지 않습니다.
    - 현재 비활성화된 기능입니다.

##### 수정 예정
    - 프롬프트를 request body로 이동시킵니다.
    - image_id 인자를 제외합니다.
    - 유저 식별 기능을 구현하여 image_id 인자를 대신합니다.
    - 이미지 출력을 정상화합니다.
---
### 3. download

    다운로드 기능을 제공하는 API입니다.
    image_id를 입력값으로 받아 예측이 완료된 이미지를 반환합니다.

##### 에러
    - 현재 비활성화된 기능입니다.

##### 수정 예정
    - image_id 인자를 제외합니다.
    - 유저 식별 기능을 구현하여 image_id 인자를 대신합니다.
---
### 4. feedback

    피드백 기능을 제공하는 API입니다.
    image_id를 입력값으로 받아 유저의 피드백을 저장합니다.

##### 에러
    - 현재 비활성화된 기능입니다.

##### 수정 예정
    - 유저 식별 기능을 구현하여 image_id 인자를 대신합니다.