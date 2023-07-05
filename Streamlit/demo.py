import streamlit as st
import zipfile
import os
import shutil
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates



def is_folder_empty(folder_path):
    file_list = os.listdir(folder_path)
    return len(file_list) == 0



def label_input1():
    example_text1 = "ex) building,tree"
    example_input1 = example_text1.split(',')

    user_input = st.text_input("라벨들을 입력하세요", value=example_text1)
    user_input = user_input.replace(', ', ',').split(',')

    st.write("입력된 라벨들")
    if user_input != example_input1:
        st.text(user_input)



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





def main():
    st.markdown("""
        <h1 style='text-align: center;'>Data Annotation Tool</h1>
        <h3 style='text-align: right;'>CV-09</h3>
    """, unsafe_allow_html=True)

    st.subheader("DEMO")

    img_file = st.file_uploader("사진 한장 업로드 하세요")

    label_input1()

    if img_file is not None:
        image = Image.open(img_file)

        st.image(image)

        if "points" not in st.session_state:
            st.session_state["points"] = []

        def get_ellipse_coords(point: tuple[int, int]) -> tuple[int, int, int, int]:
            center = point
            radius = 10
            return (
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            )


        with st.echo("below"):
            with Image.open(img_file) as img:
                draw = ImageDraw.Draw(img)

                for point in st.session_state["points"]:
                    coords = get_ellipse_coords(point)
                    draw.ellipse(coords, fill="red")

                value = streamlit_image_coordinates(img, key="pil")

                if value is not None:
                    point = value["x"], value["y"]

                    if point not in st.session_state["points"]:
                        st.session_state["points"].append(point)
                        st.experimental_rerun()

        st.write(value)
            


    st.subheader("Annotation Tool")

    option = st.radio('Select one:', ["Window", "MAC"])

    if option == "Window":
        zip_file = st.file_uploader("사진 데이터 파일 .zip 형식으로 업로드(윈도우용)", type=["zip"])
        
        if zip_file is not None and is_folder_empty("images_win"):
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall("images_win")
    
    else:
        zip_file = st.file_uploader("사진 데이터 파일 .zip 형식으로 업로드(Mac용)", type=["zip"])
        
        if zip_file is not None and is_folder_empty("images_mac"):
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall("images_mac")
            
            macosx_folder = "images_mac/__MACOSX"
            if os.path.exists(macosx_folder) and os.path.isdir(macosx_folder):
                shutil.rmtree(macosx_folder)


            image_folder = "images_mac"
            folder_list = [folder for folder in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, folder))]

            for folder in folder_list:
                folder_path = os.path.join(image_folder, folder)
                files = os.listdir(folder_path)
                for file in files:
                    file_path = os.path.join(folder_path, file)
                    shutil.move(file_path, image_folder)
                os.rmdir(folder_path)


    label_input2()

    clicked = st.button("사용 완료")

    if clicked:
        if not is_folder_empty("images_win"):
            delete_folder("images_win")
        elif not is_folder_empty("images_mac"):
            delete_folder("images_mac")


    st.subheader("Comment")
    user_comment = st.text_area("건의사항 있으시면 작성해주세요")





if __name__ == "__main__":
    main()