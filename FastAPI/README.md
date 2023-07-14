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
    zip_upload, segment, remove API 모두 정상적으로 동작합니다.

### 1. zip_upload

    zip 파일 업로드 기능을 제공하는 API입니다.
    zip 파일을 입력값으로 받아 지정된 경로에 zip 파일을 저장하고 파일의 압축을 해제합니다. 프론트엔드에서 사용자에게 ID값을 받고, datetime 모듈을 이용하여 파일 업로드 경로를 차별화합니다.

##### 수정 예정
    - 기능 구현이 완료되었습니다.
---
### 2. segment

    Segment 기능을 제공하는 API입니다.
    프론트엔드에서 사용자가 트리거를 보내면 Segment Everything을 이용하여 원본 이미로부터 객체를 Segment하고, 결과를 저장한 후 반환합니다. 사용자가 zip_upload에서 입력한 ID값을 이용하여 저장 경로를 차별화합니다.

##### 수정 예정
    - 미정
---
### 3. json_download
    
    json 파일 다운로드 기능을 제공하는 API입니다.
    프론트엔드에서 사용자가 트리거를 보내면 json 파일을 보내 다운로드가 가능하도록 합니다.

##### 수정 예정
    - Mask를 RLE로 바꾸는 기능이 구현되면 return 값을 수정할 예정입니다. 현재는 test_json 파일을 return 합니다.

---
### 4. remove

    jpg 파일 삭제 기능을 제공하는 API입니다.
    프론트엔드에서 사용자가 트리거를 보내면 사용자가 업로드한 파일들을 삭제합니다.

##### 수정 예정
    - 기능 구현이 완료되었습니다.
    - 추후 zip 파일과 json 파일을 보존하는 방향으로 바뀔 수 있습니다.