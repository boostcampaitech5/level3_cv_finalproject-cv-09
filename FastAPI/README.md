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

    현재 3개의 API를 제공하고 있습니다.
    zip_upload, segment API는 정상적으로 동작합니다.
    remove API는 정상적으로 동작하지 않습니다.

### 1. zip_upload

    zip 파일 업로드 기능을 제공하는 API입니다.
    zip 파일을 입력값으로 받아 지정된 경로에 zip 파일을 저장하고 파일의 압축을 해제합니다. 프론트엔드에서 사용자에게 ID값을 받고, datetime 모듈을 이용하여 파일 업로드 경로를 차별화합니다.

##### 수정 예정
    - 기능 구현이 완료되었습니다.
---
### 2. segment

    Segment 기능을 제공하는 API입니다.
    프론트엔드에서 사용자가 트리거를 보내면 Segment Everything을 이용하여 원본 이미로부터 객체를 Segment하고, 결과를 저장한 후 반환합니다.

##### 수정 예정
    - original_list와 segment_list가 동일한 경우의 output을 수정합니다.
    - datetime 모듈을 이용하여 소요 시간을 측정합니다
    - 적절한 파일을 출력할 수 있도록 출력 경로를 수정합니다.
---
### 3. remove

    jpg 파일 삭제 기능을 제공하는 API입니다.
    프론트엔드에서 사용자가 트리거를 보내면 jpg 파일들을 삭제합니다.

##### 에러
    - 현재 비활성화된 기능입니다.

##### 수정 예정
    - 프론트엔드 구현에 맞추어 기능을 제공할 예정입니다.