import streamlit as st
import zipfile
import os
import shutil
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

# !pip install streamlit-image-coordinates

def is_folder_empty(folder_path):
    file_list = os.listdir(folder_path)
    return len(file_list) == 0



def label_input2():
    example_text2 = "ex) human,car,road"
    example_input2 = example_text2.split(',')

    user_input = st.text_input("라벨들을 입력하세요", value=example_text2)
    user_input = user_input.replace(', ', ',').split(',')

    st.write("입력된 라벨들")
    if user_input != example_input2:
        st.text(user_input)



def delete_folder(path):
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path,file)
        os.remove(file_path)





def show_tool_page():

    st.header("Annotation Tool")

    option = st.radio('Select one:', ["Window", "MAC"])

    if option == "Window":
        zip_file = st.file_uploader("사진 데이터 파일 .zip 형식으로 업로드(윈도우용)", type=["zip"])
        if st.button('Send'):
            files = {'files' : zip_file}
            res = requests.post("http://localhost:30008/upload/window", files=files)
            if res.status_code == 200:
                st.write('Done!')
    
    else:
        zip_file = st.file_uploader("사진 데이터 파일 .zip 형식으로 업로드(Mac용)", type=["zip"])
        if st.button('Send'):
            files = {'files' : zip_file}
            res = requests.post("http://localhost:30008/upload/mac", files=files)
            if res.status_code == 200:
                st.write('Done!')


    label_input2()

    # clicked = st.button("사용 완료")

    # if clicked:
    #     if not is_folder_empty("images_win"):
    #         delete_folder("images_win")
    #     elif not is_folder_empty("images_mac"):
    #         delete_folder("images_mac")
