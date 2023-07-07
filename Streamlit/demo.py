import streamlit as st
import torch
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
from typing import Tuple
from msam import wrapper_msam


def label_input1():
    example_text1 = "ex) building,tree"
    example_input1 = example_text1.split(',')

    user_input = st.text_input("라벨들을 입력하세요", value=example_text1)
    user_input = user_input.replace(', ', ',').split(',')

    st.write("입력된 라벨들")
    if user_input != example_input1:
        st.text(user_input)


def get_ellipse_coords(point: Tuple[int, int]) -> Tuple[int, int, int, int]:
            center = point
            radius = 8
            return (
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            )



def show_demo_page():
    st.header("DEMO")

    img_file = st.file_uploader("사진 한장 업로드 하세요")

    label_input1()

    if img_file is not None:
        image = Image.open(img_file)

        st.image(image, width=600)

        if "points" not in st.session_state:
            st.session_state["points"] = []
        
        draw = ImageDraw.Draw(image)

        for point in st.session_state["points"]:
            coords = get_ellipse_coords(point)
            draw.ellipse(coords, fill="red")


        value = streamlit_image_coordinates(image, width=600)

        if value is not None:
            point = value["x"] * (float(image.size[0]) / 600), (value["y"] * (image.size[0]/600))

            st.text(point)
            if point not in st.session_state["points"]:
                st.session_state["points"].append(point)
                st.experimental_rerun()

