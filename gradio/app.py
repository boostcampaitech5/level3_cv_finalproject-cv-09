import os
import gradio as gr
import numpy as np
import io
from PIL import Image
from zipfile import ZipFile
import requests


def zip_upload(file_obj, id):
    with ZipFile(file_obj.name, "r") as f:
        f.extractall(f"data/{id}")
    data = {"id": str(id)}
    with open(file_obj.name, "rb") as f:
        files = {"files": f}
        res = requests.post(
            "http://115.85.182.123:30008/zip_upload/",
            data=data,
            files=files,
        )
    return os.listdir(f"data/{id}")


def start_annotation(img_list):
    global img_iter
    img_iter = iter(img_list)
    return next(img_iter)


def next_img():
    global img_iter
    return next(img_iter)


def viz_img(id, path):
    personal_path = f"{id}/{path}"
    return Image.open(os.path.join("data", personal_path)), segment(id, path)


def segment(id, img_path):
    data = {"path": os.path.join(str(id), str(img_path))}
    seg = requests.post("http://115.85.182.123:30008/segment/", data=data)
    return Image.open(io.BytesIO(seg.content))


def remove(id):
    data = {"id": str(id)}
    res = requests.post("http://115.85.182.123:30008/remove/", data=data)
    return res.status_code


# Description
title = "<center><strong><font size='8'>Faster Segment Anything(MobileSAM)<font></strong></center>"

description_e = """This is a demo of [Faster Segment Anything(MobileSAM) Model](https://github.com/ChaoningZhang/MobileSAM).
                   We will provide box mode soon. 
                   Enjoy!
                
              """

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


def get_points(image, evt: gr.SelectData):
    x, y = evt.index[0], evt.index[1]
    pixels = image.load()
    return x, y, pixels[x, y]


cond_img_e = gr.Image(label="Input", type="pil")
segm_img_e = gr.Image(label="Mobile SAM Image", interactive=False, type="pil")
id = gr.Textbox()
img_list = gr.JSON()
img_iter = None  # ["img1.jpg", .... ]
# next(img_iter)

grounding_dino_SAM_img_e = gr.Image(
    label="Clip_Segmentation Image", interactive=False, image_mode="RGBA"
)

with gr.Blocks(css=css, title="Faster Segment Anything(MobileSAM)") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(title)
    with gr.Tab("file upload Tab"):
        gr.Interface(
            zip_upload, inputs=["file", id], outputs=img_list, allow_flagging="never"
        )
        label_list = gr.Textbox(interactive=True)
        with gr.Row():
            set_label_btn_e = gr.Button("Set Entire label", size="sm")
            start_btn_e = gr.Button("Start Annotation", size="sm")
    with gr.Tab("Annotation Tab"):
        present_img = gr.Textbox(interactive=False)
        cond_img_e.render()
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
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

    next_btn_e.click(next_img, outputs=present_img)
    request_btn_e.click(remove, inputs=[id])

    segm_img_e.select(get_points, inputs=[segm_img_e], outputs=[coord_value])

    set_label_btn_e.click(
        fn=lambda value: label_checkbox.update(
            choices=value.replace(", ", ",").split(",")
        ),
        inputs=label_list,
        outputs=label_checkbox,
    )
    start_btn_e.click(
        fn=start_annotation,
        inputs=img_list,
        outputs=present_img,
    )
    present_img.change(
        fn=viz_img, inputs=[id, present_img], outputs=[cond_img_e, segm_img_e]
    )

demo.queue()
demo.launch()
