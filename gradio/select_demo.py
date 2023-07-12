import gradio as gr
import numpy as np
import torch
import io
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from utils.tools_gradio import fast_process

from PIL import Image
from zipfile import ZipFile
import requests

colors = [
    (0, 0, 0),
    (0.8196078431372549, 0.2901960784313726, 0.25882352941176473),
    (0.42745098039215684, 0.9490196078431372, 0.2),
    (0.9490196078431372, 0.9254901960784314, 0.8862745098039215),
    (0.5764705882352941, 0.19607843137254902, 0.6235294117647059),
    (0.0196078431372549, 0.41568627450980394, 0.9725490196078431),
    (0.3764705882352941, 0.20784313725490197, 0.09411764705882353),
    (0.12156862745098039, 0.4745098039215686, 0.38823529411764707),
    (0.00392156862745098, 0.34901960784313724, 0.01568627450980392),
    (0.4470588235294118, 0.00392156862745098, 0.03137254901960784),
    (0.32941176470588235, 0.34901960784313724, 0.7607843137254902),
]


# def zip_to_json(file_obj):
#     files = []
#     print(file_obj.name)
#     with ZipFile(file_obj.name) as zfile:
#         for zinfo in zfile.infolist():
#             files.append(
#                 zinfo.filename,
#             )
#         print(zfile)
#     return files



def zip_upload(file_obj, id):
    #
    # f = ZipFile(file_obj.name, "r")
    # with ZipFile(file_obj.name) as zfile:
    #     files = {"files": zfile}
    data = {"id": str(id)}
    with open(file_obj.name, "rb") as f:
        files = {"files": f}
        res = requests.post(
            "http://115.85.182.123:30008/zip_upload/", data=data, files=files,
        )
    return res.status_code

def segment():
    res = requests.get("http://115.85.182.123:30008/segment/")
    return Image.open(io.BytesIO(res.content))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mobile_Sam load code
sam_checkpoint = "weights/mobile_sam.pt"
model_type = "vit_t"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam = mobile_sam.to(device=device)
mobile_sam.eval()

mask_generator = SamAutomaticMaskGenerator(mobile_sam)
predictor = SamPredictor(mobile_sam)

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


@torch.no_grad()
def segment_everything(
    image,
    input_size=1024,
    better_quality=True,
    use_retina=True,
    mask_random_color=True,
):
    global mask_generator

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    nd_image = np.array(image)
    annotations = mask_generator.generate(nd_image)

    fig = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
    )
    return fig


def get_points(image, evt: gr.SelectData):
    x, y = evt.index[0], evt.index[1]
    pixels = image.load()
    return x, y, pixels[x, y]


cond_img_e = gr.Image(label="Input", value=default_example[0], type="pil")
segm_img_e = gr.Image(label="Mobile SAM Image", interactive=False, type="pil")
id = gr.Textbox()
clipseg_img_e = gr.Image(
    label="Clip_Segmentation Image", interactive=True, image_mode="RGBA"
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
            # Title
            gr.Markdown(title)
    with gr.Tab("file upload Tab"):
        gr.Interface(zip_upload, inputs=["file", id], outputs="text")
        label_list = gr.Textbox(interactive=True)
    with gr.Tab("Everything mode"):
        # Images
        cond_img_e.render()
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                input_size_slider.render()
                segment_btn_e = gr.Button("Segment Everything", variant="primary")
            with gr.Column(scale=1):
                label_checkbox = gr.CheckboxGroup(
                    choices=[],
                    value=["swam", "slept"],
                    label="select label in present image",
                    interactive=True,
                )
                clipseg_btn_e = gr.Button("clip_segmentation", variant="primary")
        # Submit & Clear
        with gr.Row():
            with gr.Column(scale=1):
                segm_img_e.render()
                with gr.Row():
                    add_btn_e = gr.Button("add", variant="secondary")
                    delete_btn_e = gr.Button("delete", variant="secondary")

            with gr.Column():
                clipseg_img_e.render()
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

    segment_btn_e.click(
        segment,
        outputs=[segm_img_e]
    )
    
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

"""
이미지를 원본이랑 마스크로 분리, 
modified_img_e 이미지 select 될때 마우스 좌표를 
마스크 이미지에 적용, 해당 픽셀의 값 반환 
"""
