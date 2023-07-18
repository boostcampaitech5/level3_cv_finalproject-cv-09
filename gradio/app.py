import os
import gradio as gr
import numpy as np
import io
from PIL import Image
from zipfile import ZipFile
import requests


def alert_eof():
    message = f"마지막 이미지입니다."
    # JavaScript를 사용하여 alert 창을 생성하고 메시지를 출력
    alert_script = f"alert('{message}');"
    display(HTML(f"<script>{alert_script}</script>"))


def zip_upload(img_zip, id):
    with ZipFile(img_zip.name, "r") as f:
        f.extractall(f"data/{id}")
    data = {"id": str(id)}
    # with open(img_zip.name, "rb") as f:
    #     files = {"files": f}
    #     res = requests.post(
    #         "http://115.85.182.123:30008/zip_upload/",
    #         data=data,
    #         files=files,
    #     )
    with open(img_zip.name, "rb") as f:
        files = {"files": f}
        res = requests.post(
            "http://118.67.142.203:30008/zip_upload/",
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
    try:
        next_img = next(img_iter)
    except StopIteration:
        gr.Interface(fn=alert_eof, outputs=None).launch()
    return next_img


def viz_img(id, path):
    personal_path = f"{id}/{path}"
    return Image.open(os.path.join("data", personal_path))


def segment(id, img_path):
    data = {"path": os.path.join(str(id), str(img_path))}
    seg = requests.post("http://118.67.142.203:30008/segment/", data=data)
    return Image.open(io.BytesIO(seg.content))


def segment_text(id, img_path, text_prompt):
    string_prompt = ' . '.join(text_prompt)
    data = {"path": os.path.join(str(id), str(img_path)), "text_prompt": string_prompt}
    print(data)
    seg = requests.post("http://118.67.142.203:30008/segment_text/", data=data)
    return Image.open(io.BytesIO(seg.content))


def segment_reqest(id, img_path, text_prompt):
    return segment(id, img_path), segment_text(id, img_path, text_prompt)


# def remove(id):
#     data = {"id": str(id)}
#     res = requests.post("http://115.85.182.123:30008/remove/", data=data)
#     return res.status_code


# Description
title = "<center><strong><font size='8'>Image Annotation Tool<font></strong></center>"

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
    label="grounding_dino_SAM_img", interactive=False, type="pil"
)
my_theme = gr.Theme.from_hub("nuttea/Softblue")
with gr.Blocks(
    css=css, title="Faster Segment Anything(MobileSAM)", theme=my_theme
) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(title)
    with gr.Tab("file upload Tab"):
        gr.Interface(
            zip_upload, inputs=["file", id], outputs=img_list, allow_flagging="never"
        )
        label_list = gr.Textbox(
            label="Input entire label list that separated ,",
            info="ex: human, tree, car, sky",
            interactive=True,
        )
        with gr.Row():
            set_label_btn_e = gr.Button("Set Entire label", size="sm")
            start_btn_e = gr.Button("Start Annotation", size="sm")
    with gr.Tab("Annotation Tab"):
        with gr.Row():
            present_img = gr.Textbox(label="present Image name", interactive=False)
        with gr.Row():
            with gr.Column():
                cond_img_e.render()
            with gr.Column():
                with gr.Tab("Grounding Dino"):
                    grounding_dino_SAM_img_e.render()
                with gr.Tab("Segment Everything"):
                    segm_img_e.render()
        with gr.Row(variant="panel"):
            label_checkbox = gr.CheckboxGroup(
                choices=[],
                label="select label in present image",
                interactive=True,
            )

        with gr.Row():
            with gr.Column():
                add_btn_e = gr.Button("add", variant="secondary")
            with gr.Column():
                delete_btn_e = gr.Button("delete", variant="secondary")
            with gr.Column():
                next_btn_e = gr.Button("next", variant="secondary")
            with gr.Column():
                request_btn_e = gr.Button("request", variant="primary")
        with gr.Row():
            coord_value = gr.Textbox(label="Preview annotation.json")

    ################ 1 page buttons ################
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
    ################################################

    ################ 2 page buttons ################
    next_btn_e.click(next_img, outputs=present_img)
    request_btn_e.click(
        segment_reqest,
        inputs=[id, present_img, label_checkbox],
        outputs=[segm_img_e, grounding_dino_SAM_img_e],
    )
    ################################################

    segm_img_e.select(get_points, inputs=[segm_img_e], outputs=[coord_value])

    present_img.change(
        fn=viz_img,
        inputs=[id, present_img],
        outputs=[cond_img_e],
    )
demo.queue()
demo.launch()
