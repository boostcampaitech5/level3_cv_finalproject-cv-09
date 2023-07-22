import os
import gradio as gr
import numpy as np
import io
from PIL import Image
from zipfile import ZipFile
import requests
from collections import deque
import shutil
import torch
import json
import sys
from torchvision.utils import draw_segmentation_masks
import time
import asyncio


def draw_image(image, masks, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if len(masks) > 0:
        image = draw_segmentation_masks(
            image, masks=masks, colors=["cyan"] * len(masks), alpha=alpha
        )
    return image.numpy().transpose(1, 2, 0)

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def zip_upload(is_drive, img_zip, id):
    with ZipFile(img_zip.name, "r") as f:
        f.extractall(f"data/{id}")
    data = {"id": str(id)}
    with open(img_zip.name, "rb") as f:
        files = {"files": f}
        res = requests.post(
            "http://118.67.142.203:30008/zip_upload/",
            data=data,
            files=files,
        )
    return os.listdir(f"data/{id}")


def start_annotation(img_list):
    global img_deque
    img_deque = deque(img_list)
    return img_deque[0]


def next_img():
    img_deque.rotate(-1)
    return img_deque[0]


def prev_img():
    img_deque.rotate(1)
    return img_deque[0]


def viz_img(id, path):
    personal_path = f"{id}/{path}"
    return Image.open(os.path.join("data", personal_path))


def segment(id, img_path):
    data = {"path": os.path.join(str(id), str(img_path))}
    seg = requests.post("http://118.67.142.203:30008/segment/", data=data)
    mask_dict = json.loads(seg.content)
    sam_masks = list()
    for idx, mask in enumerate(mask_dict["masks"]):
        seg_mask = rle_decode(mask, mask_dict['size'])
        sam_masks.append(np.array(seg_mask))
    print(np.array(sam_masks[0]).shape)
    print(len(sam_masks))
    return sam_masks
    # return Image.open(io.BytesIO(seg.content))


def hrnet_request(id, img_path):
    data = {"path": os.path.join(str(id), str(img_path))}
    res = requests.post("http://118.67.142.203:30008/segment_hrnet/", data=data)
    # Please check if next code works
    hrnet_img, hrnet_json = res.content
    return Image.open(io.BytesIO(hrnet_img))
    
async def make_annotation_json(img_path, data):
    global annotation_info
    annotation_dict = dict()
    annotation_dict['image_path'] = img_path
    annotation_dict['size'] = data['size']
    annotation_dict['masks'] = list()
    for label, mask in data['masks'].items():
        mask_info = {"label" : label, "mask" : mask}
        annotation_dict['masks'].append(mask_info)
    annotation_info['annotation'].append(annotation_dict)

# 현석이가 만들어 줄 것.
def segment_text(id, img_path, text_prompt, threshold):
    global annotation_info
    string_prompt = " . ".join(text_prompt)
    img_prefix = f"data/{id}"
    image_pil = Image.open(os.path.join(img_prefix, img_path)).convert("RGB")
    data = {"path": os.path.join(str(id), str(img_path)), "text_prompt": string_prompt, "threshold": threshold}
    seg = requests.post("http://118.67.142.203:30008/segment_text/", data=data)
    mask_dict = json.loads(seg.content)
    temp = []
    asyncio.run(make_annotation_json(img_path, mask_dict))
    for label, mask in mask_dict["masks"].items():
        seg_mask = rle_decode(mask, mask_dict['size'])
        temp.append((np.array(seg_mask), label))
    return image_pil, temp


def segment_request(id, threshold, img_path, text_prompt):
    return segment(id, img_path), segment_text(
        id, img_path, text_prompt, threshold
    )  # 얘 output이 [(building_image1, "buildings1"), (building_image2, "buildings2")] 이런식으로 나와야함


def json_download(id, img_path):
    data = {"path": os.path.join(str(id), str(img_path))}
    res = requests.post("http://118.67.142.203:30008/json_download/", data=data)
    return res.content


def finish(id):
    data = {"id": str(id)}
    res = requests.post("http://118.67.142.203:30008/remove/", data=data)
    shutil.rmtree(f"data/{str(id)}")  # 확인 필요

def save_annotation(id):
    global annotation_info
    file_prefix = os.join("data", id)
    file_path = os.join(file_prefix, "data.json")  # 저장할 파일 경로 및 이름
    annotation_info['user_id'] = id
    with open(file_path, "w") as json_file:
        json.dump(annotation_info, json_file, indent=4) 
    
    
# Description
title = "<center><strong><font size='8'>Image Annotation Tool<font></strong></center>"

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


def get_points(image, evt: gr.SelectData):
    x, y = evt.index[0], evt.index[1]
    pixels = image.load()
    return x, y, pixels[x, y]


cond_img_e = gr.Image(label="Input", type="pil")
segm_img_e = gr.Image(label="Mobile SAM", interactive=False, type="pil")
gdSAM_img_e = gr.AnnotatedImage(label="GDSAM", interactive=False)

id = gr.Textbox()
img_list = gr.JSON()
annotation_info = {"annotation" : list()}
'''
user_id : string
annotation : array of dict
ㄴimage_path : string
ㄴsize : array of int
ㄴmasks : array of dict
    ㄴlabel : string
    ㄴmask : string (rle-encoded)
'''
drive_data = gr.Radio(["drive dataset", "others"])
my_theme = gr.Theme.from_hub("nuttea/Softblue")
with gr.Blocks(
    css=css, title="Faster Segment Anything(MobileSAM)", theme=my_theme
) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(title)
    with gr.Tab("file upload Tab"):
        gr.Interface(
            zip_upload,
            inputs=[drive_data, "file", id],
            outputs=img_list,
            allow_flagging="never",
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
                with gr.Tab("Original Image"):
                    cond_img_e.render()
            with gr.Column():
                with gr.Tab("Grounding Dino"):
                    gdSAM_img_e.render()
                with gr.Tab("Segment Everything"):
                    segm_img_e.render()
        with gr.Row():
            threshold = gr.Slider(0.01, 1, value=0.3, step=0.01, label="Threshold", interactive = True, info="Choose between 0.01 and 1")
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
                prev_btn_e = gr.Button("prev", variant="secondary")
            with gr.Column():
                next_btn_e = gr.Button("next", variant="secondary")
            with gr.Column():
                request_btn_e = gr.Button("request", variant="primary")
        with gr.Row():
            coord_value = gr.Textbox(label="Preview annotation.json")
        with gr.Row():
            with gr.Column():
                finish_btn_e = gr.Button("Finish")
            with gr.Column():
                save_btn_e = gr.Button("Save")

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
    prev_btn_e.click(prev_img, outputs=present_img)
    next_btn_e.click(next_img, outputs=present_img)
    request_btn_e.click(
        segment_request,
        inputs=[id, threshold, present_img, label_checkbox],
        outputs=[
            segm_img_e,
            gdSAM_img_e,  # gdSAM얘는 [(building_image, "buildings")] 이런식으로 들어가야함.
        ],
    )
    finish_btn_e.click(finish, inputs=id)
    save_btn_e.click(save_annotation, inputs=id)
    ################################################

    segm_img_e.select(get_points, inputs=[segm_img_e], outputs=[coord_value])

    present_img.change(
        fn=viz_img,
        inputs=[id, present_img],
        outputs=[cond_img_e],
    )
demo.queue()
demo.launch()
