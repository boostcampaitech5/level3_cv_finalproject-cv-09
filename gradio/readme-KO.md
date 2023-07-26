# Data Annotation Tool - FE(Gradio) Server

## 1. Install requirements

```
pip install -r requirements.txt
```

## 2. run `app.py`

```
gradio run app.py
```

if you run `app.py` externel server

```
# in app.py, fixed last line code
demo.launch(server_name="0.0.0.0", server_port=your port)
```

## 3. Just use the tool

![image1](assets/image1.png)

1. 데이터셋 유형을 확인하세요 (현재는 걷기 데이터셋만 지원됩니다).
2. 이미지.zip 파일을 업로드하세요.
3. 사용자 ID를 입력하세요.
4. "제출하기" 버튼을 클릭하세요.
   (zip 파일에서 파일 목록을 확인할 수 있습니다).
5. "Set Entire label", "Start Annotation"을 클릭하세요.
6. 주석 탭으로 이동하세요.

![image2](assets/image2.png)

1. 먼저 원본 이미지를 확인할 수 있습니다. 유사 라벨을 얻으려면 요청 버튼을 클릭하세요.
2. 테이블 이미지 탭에서 유사 라벨 이미지를 얻을 수 있습니다.
3. 라벨을 수정하려면 "Add Label"을 체크하고 드롭다운에서 라벨을 선택한 후, 이미지에서 변경하고자 하는 부분을 클릭하고 수정 버튼을 누르세요.
4. 이전 버튼과 다음 버튼을 사용하여 다음 이미지 또는 이전 이미지로 이동할 수 있습니다.
5. 현재 이미지의 주석을 저장하려면 저장 버튼을 누르세요.
6. 주석 작업을 완료하고 전체 주석 파일을 다운로드하려면 완료 버튼을 누르세요.
