import os
import gradio as gr
import numpy as np
import io

from PIL import Image
from zipfile import ZipFile
import requests


def zip_upload(file_obj, id):
    with ZipFile(file_obj, "r") as f:
        f.extractall(f"temp_zip/{id}")
    data = {"id": str(id)}
    with open(file_obj.name, "rb") as f:
        files = {"files": f}
        res = requests.post(
            "http://115.85.182.123:30008/zip_upload/",
            data=data,
            files=files,
        )
    return os.listdir(f"temp_zip/{id}")


def segment(id, img_path):
    data = {"path": os.path.join(str(id), str(img_path))}
    res = requests.post("http://115.85.182.123:30008/segment/", data=data)
    return Image.open(io.BytesIO(res.content))


def remove():
    res = requests.get("http://115.85.182.123:30008/remove/")
    return res.status_code


# Description
title = "<center><strong><font size='8'>Faster Segment Anything(MobileSAM)<font></strong></center>"

description_e = """This is a demo of [Faster Segment Anything(MobileSAM) Model](https://github.com/ChaoningZhang/MobileSAM).

                   We will provide box mode soon. 

                   Enjoy!
                
              """


examples = [
    ["images/N-B-P-004_000433.jpg"],
    ["images/N-B-P-004_017137.jpg"],
    ["images/N-B-P-021_000109.jpg"],
    ["images/N-E-C-020_000505.jpg"],
    ["images/N-E-C-020_002305.jpg"],
    ["images/S-W-P-004_015841.jpg"],
]

default_example = examples[0]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


def get_points(image, evt: gr.SelectData):
    x, y = evt.index[0], evt.index[1]
    pixels = image.load()
    return x, y, pixels[x, y]


cond_img_e = gr.Image(label="Input", type="pil")
segm_img_e = gr.Image(label="Mobile SAM Image", interactive=False, type="pil")
id = gr.Textbox()
grounding_dino_SAM_img_e = gr.Image(
    label="Clip_Segmentation Image", interactive=False, image_mode="RGBA"
)

input_size_slider = gr.components.Slider(
    minimum=512,
    maximum=1024,
    value=1024,
    step=64,
    label="Input_size",
    info="Our model was trained on a size of 1024",
)

with gr.Blocks(css=css, title="Faster Segment Anything(MobileSAM)") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(title)
    with gr.Tab("file upload Tab"):
        gr.Interface(zip_upload, inputs=["file", id], outputs="text")
        label_list = gr.Textbox(interactive=True)
    with gr.Tab("Annotation Tab"):
        cond_img_e.render()
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                input_size_slider.render()
                segment_btn_e = gr.Button("Segment Everything", variant="primary")
            with gr.Column(scale=1):
                label_checkbox = gr.CheckboxGroup(
                    choices=[],
                    label="select label in present image",
                    interactive=True,
                )
                clipseg_btn_e = gr.Button("clip_segmentation", variant="primary")
        with gr.Row():
            with gr.Tab("Grounding Dino"):
                grounding_dino_SAM_img_e.render()
            with gr.Tab("Segment Everything"):
                segm_img_e.render()

        with gr.Row():
            with gr.Column():
                add_btn_e = gr.Button("add", variant="secondary")
            with gr.Column():
                delete_btn_e = gr.Button("delete", variant="secondary")
            with gr.Column():
                next_btn_e = gr.Button("next", variant="secondary")
            with gr.Column():
                request_btn_e = gr.Button("request", variant="secondary")
        with gr.Row():
            coord_value = gr.Textbox()
    # segment_btn_e.click(
    #     segment_everything,
    #     inputs=[
    #         cond_img_e,
    #         input_size_slider,
    #     ],
    #     outputs=[segm_img_e],
    # )

    segment_btn_e.click(segment, inputs=[], outputs=[segm_img_e])

    # next_btn_e.click(

    # )

    request_btn_e.click(remove)

    segm_img_e.select(get_points, inputs=[segm_img_e], outputs=[coord_value])
    # clipseg_btn_e.click(
    #     clip_segmentation, inputs=[cond_img_e, label_checkbox], outputs=[clipseg_img_e]
    # )
    label_list.change(
        fn=lambda value: label_checkbox.update(
            choices=value.replace(", ", ",").split(",")
        ),
        inputs=label_list,
        outputs=label_checkbox,
    )
demo.queue()
demo.launch()
