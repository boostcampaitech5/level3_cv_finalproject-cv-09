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

np.set_printoptions(threshold=784, linewidth=np.inf)
save_annotate = []
hrnet_label = [
    "",
    "background,wheelchair,truck,traffic_sign,traffic_light,stroller,stop,scooter,pole,person,motorcycle,dog,cat,carrier,car,bus,bollard,bicycle,barricade",
]
classes = {
    "background": [0, (255, 255, 255)],  # 배경
    "wheelchair": [1, (255, 0, 0)],  # 휠체어
    "carrier": [2, (0, 64, 0)],  # 화물차
    "stop": [3, (0, 255, 255)],  # 정지선
    "cat": [4, (64, 0, 0)],  # 고양이
    "pole": [5, (0, 128, 128)],  # 대
    "traffic_light": [6, (255, 0, 255)],  # 신호등
    "traffic_sign": [7, (0, 0, 255)],  # 교통 표지판
    "stroller": [8, (255, 255, 0)],  # 유모차
    "dog": [9, (255, 128, 255)],  # 개
    "barricade": [10, (0, 192, 0)],  # 바리케이드
    "person": [11, (128, 0, 128)],  # 사람
    "scooter": [12, (128, 128, 0)],  # 스쿠터
    "car": [13, (0, 0, 64)],  # 차
    "truck": [14, (0, 255, 0)],  # 트럭
    "bus": [15, (64, 64, 0)],  # 버스
    "bollard": [16, (64, 0, 64)],  # 인도 블럭 바리케이드 비슷한거
    "motorcycle": [17, (128, 0, 255)],  # 오토바이
    "bicycle": [18, (0, 64, 64)],  # 자전거
}
color_dict = {
    "human": "#000000",
    "wheelchair": "#ff0000",
    "carrier": "#004000",
    "stop": "#00ffff",
    "cat": "#400000",
    "pole": "#008080",
    "traffic_light": "#ff00ff",
    "traffic_sign": "#0000ff",
    "stroller": "#ffff00",
    "dog": "#ff80ff",
    "barricade": "#00c000",
    "person": "#800080",
    "scooter": "#808000",
    "car": "#000040",
    "truck": "#00ff00",
    "bus": "#404000",
    "bollard": "#400040",
    "motorcycle": "#8000ff",
    "bicycle": "#004040",
}


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
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
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
            "http://127.0.0.1:40001/zip_upload/",
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
    seg = requests.post("http://127.0.0.1:40001/segment/", data=data)
    mask_dict = json.loads(seg.content)

    return mask_dict


def list_to_image_1channel(data):
    # 2차원 리스트의 크기를 구합니다.
    height = len(data)
    width = len(data[0])

    # 흑백 이미지 모드인 'L' 모드로 새 이미지 객체를 생성합니다.
    image = Image.new("L", (width, height))

    # 2차원 리스트의 값을 이미지의 픽셀로 복사합니다.
    for y in range(height):
        for x in range(width):
            # 이미지의 픽셀은 0부터 255의 값을 가지며, 0은 검은색, 255는 흰색을 의미합니다.
            pixel_value = data[y][x]
            image.putpixel((x, y), pixel_value)

    return image


def hrnet_request(id, img_path):
    img_prefix = f"data/{id}"
    image_pil = Image.open(os.path.join(img_prefix, img_path)).convert("RGB")
    data = {"path": os.path.join(str(id), str(img_path))}
    res = requests.post("http://127.0.0.1:40001/segment_hrnet/", data=data)
    # Please check if next code works
    mask_dict = json.loads(res.content)

    temp = []
    for label, mask in mask_dict["masks"].items():
        seg_mask = rle_decode(mask, mask_dict["size"])
        # 2차원 리스트를 흑백 이미지로 변환합니다.
        # image = list_to_image_1channel(mask)
        # image.save("output.png")
        temp.append((seg_mask, label))
    global save_annotate
    save_annotate = temp
    return image_pil, temp


async def make_annotation_json(img_path, data):
    global annotation_info
    annotation_dict = dict()
    print(data)
    annotation_dict["image_path"] = img_path
    annotation_dict["size"] = data["size"]
    annotation_dict["masks"] = list()
    for label, mask in data["masks"].items():
        mask_info = {"label": label, "mask": mask}
        annotation_dict["masks"].append(mask_info)
    annotation_info["annotation"].append(annotation_dict)


# 현석이가 만들어 줄 것.
def segment_text(id, img_path, text_prompt, threshold):
    global annotation_info
    string_prompt = " . ".join(text_prompt)
    img_prefix = f"data/{id}"
    image_pil = Image.open(os.path.join(img_prefix, img_path)).convert("RGB")
    data = {
        "path": os.path.join(str(id), str(img_path)),
        "text_prompt": string_prompt,
        "threshold": threshold,
    }
    seg = requests.post("http://127.0.0.1:40001/segment_text/", data=data)
    mask_dict = json.loads(seg.content)

    temp = []
    # asyncio.run(make_annotation_json(img_path, mask_dict))
    for label, mask in mask_dict["masks"].items():
        seg_mask = rle_decode(mask, mask_dict["size"])
        temp.append((seg_mask, label))

    return image_pil, temp


def segment_request(id, threshold, img_path, text_prompt):
    return segment(id, img_path), hrnet_request(
        id, img_path
    )  # 얘 output이 [(building_image1, "buildings1"), (building_image2, "buildings2")] 이런식으로 나와야함


def json_download(id, img_path):
    data = {"path": os.path.join(str(id), str(img_path))}
    res = requests.post("http://127.0.0.1:40001/json_download/", data=data)
    return res.content


def finish(id):
    data = {"id": str(id)}
    res = requests.post("http://127.0.0.1:40001/remove/", data=data)
    shutil.rmtree(f"data/{str(id)}")  # 확인 필요


def save_annotation(id):
    global annotation_info
    file_prefix = os.join("data", id)
    file_path = os.join(file_prefix, "data.json")  # 저장할 파일 경로 및 이름
    annotation_info["user_id"] = id
    with open(file_path, "w") as json_file:
        json.dump(annotation_info, json_file, indent=4)


# Description
title = "<center><strong><font size='8'>Image Annotation Tool<font></strong></center>"

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


def get_points(image, evt: gr.SelectData):
    w, h = image.size
    x, y = evt.index[0], evt.index[1]
    return x, y, w, h


def remove_mask_from_image(image_path, mask_array, output_path):
    # 이미지와 마스크를 엽니다.
    image = Image.open(image_path)
    mask = Image.fromarray(mask_array.astype("uint8") * 255, mode="L")

    # 이미지와 마스크의 크기가 같은지 확인합니다.
    if image.size != mask.size:
        raise ValueError("Image and mask size must be the same.")

    # 마스크 부분을 지우기 위해 투명한 값을 가진 새로운 이미지를 생성합니다.
    transparent_image = Image.new("RGBA", image.size, (0, 0, 0, 0))

    # 이미지와 마스크의 모든 픽셀을 탐색하며 마스크 부분을 투명한 이미지에 추가합니다.
    for x in range(image.width):
        for y in range(image.height):
            pixel = image.getpixel((x, y))
            mask_pixel = mask.getpixel((x, y))
            if mask_pixel == 0:  # 마스크 값이 0인 부분은 투명하게 만듭니다.
                transparent_image.putpixel((x, y), (0, 0, 0, 0))
            else:  # 마스크 값이 0이 아닌 부분은 원래 이미지의 값을 유지합니다.
                transparent_image.putpixel((x, y), pixel)

    # 결과 이미지를 저장합니다.
    transparent_image.save(output_path)


def merge_images_to_mask(image_path, mask_array, output_path):
    # 이미지와 마스크를 엽니다.
    image = Image.open(image_path)
    mask = Image.fromarray(mask_array.astype("uint8") * 255, mode="L")

    # 빈 채널을 만들기 위해 투명한 값을 가진 새로운 이미지를 생성합니다.
    empty_image = Image.new("RGBA", image.size, (0, 0, 0, 0))

    # 빨간색 마스크를 생성합니다.
    red_mask = Image.new("L", mask.size, 0)
    for x in range(mask.width):
        for y in range(mask.height):
            pixel = mask.getpixel((x, y))
            if pixel == 255:  # (255, 0, 0, 0)인 부분을 빨간색 마스크로 설정합니다.
                red_mask.putpixel((x, y), 255)

    # 빨간색 마스크와 이미지를 합쳐서 새로운 이미지를 생성합니다.
    merged_image = Image.merge(
        "RGBA", [empty_image, empty_image, empty_image, red_mask]
    )

    # 1체널 True, False로 이루어진 마스크를 생성합니다.
    result_mask = np.array(merged_image.convert("L")) > 0

    # 결과 마스크를 저장합니다.
    np.save(output_path, result_mask)


def add_mask(coord, sam_image, dropdown, cond_img):
    x, y, w, h = coord
    global classes
    temp = rle_decode(sam_image["masks"][0], sam_image["size"])

    prev_h, prev_w = temp.shape

    x_ratio = prev_w / w
    y_ratio = prev_h / h

    x = int(x * x_ratio)
    y = int(y * y_ratio)

    mask_idx = 0
    for idx, mask in enumerate(sam_image["masks"]):
        value = rle_decode(mask, sam_image["size"])[y, x]
        if value:
            mask_idx = idx
    print(rle_decode(sam_image["masks"][mask_idx], sam_image["size"]))
    mask_to_add = Image.fromarray(
        rle_decode(sam_image["masks"][mask_idx], sam_image["size"])
    )
    mask_to_add = mask_to_add.resize((w, h))
    output_path = "output_image.jpg"
    mask_to_add.save(output_path)

    save_annotate[classes[dropdown][0]] = list(save_annotate[classes[dropdown][0]])

    save_annotate[classes[dropdown][0]][0] = np.logical_or(
        np.array(mask_to_add), save_annotate[classes[dropdown][0]][0]
    )
    save_annotate[classes[dropdown][0]] = tuple(save_annotate[classes[dropdown][0]])

    return cond_img, save_annotate


def delete_mask(coord, sam_image, dropdown, cond_img):
    x, y, w, h = coord
    global classes
    temp = rle_decode(sam_image["masks"][0], sam_image["size"])

    prev_h, prev_w = temp.shape

    x_ratio = prev_w / w
    y_ratio = prev_h / h

    x = int(x * x_ratio)
    y = int(y * y_ratio)

    mask_idx = 0
    for idx, mask in enumerate(sam_image["masks"]):
        value = rle_decode(mask, sam_image["size"])[y, x]
        if value:
            mask_idx = idx
    print(rle_decode(sam_image["masks"][mask_idx], sam_image["size"]))
    mask_to_add = Image.fromarray(
        rle_decode(sam_image["masks"][mask_idx], sam_image["size"])
    )
    mask_to_add = mask_to_add.resize((w, h))
    output_path = "output_image.jpg"
    mask_to_add.save(output_path)

    save_annotate[classes[dropdown][0]] = list(save_annotate[classes[dropdown][0]])

    save_annotate[classes[dropdown][0]][0] = np.logical_or(
        np.logical_not(np.array(mask_to_add)), save_annotate[classes[dropdown][0]][0]
    )
    save_annotate[classes[dropdown][0]] = tuple(save_annotate[classes[dropdown][0]])

    return cond_img, save_annotate


cond_img_e = gr.Image(label="Input", interactive=False, type="pil")
# segm_img_e = gr.Image(label="Mobile SAM", interactive=False, type="pil")
gdSAM_img_e = gr.AnnotatedImage(label="GDSAM", interactive=True, color_map=color_dict)
# HRNet_img_e = gr.AnnotatedImage(label="HRNet", interactive=False)

# temp = [HRNet_img_e, gdSAM_img_e]
id = gr.Textbox()
img_list = gr.JSON()

annotation_info = {"annotation": list()}
"""
user_id : string
annotation : array of dict
ㄴimage_path : string
ㄴsize : array of int
ㄴmasks : array of dict
    ㄴlabel : string
    ㄴmask : string (rle-encoded)
"""
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
            value=hrnet_label[1],
            info="ex: human, tree, car, sky",
            interactive=True,
            visible=False,
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
                    coord = gr.JSON()
            with gr.Column():
                # with gr.Tab("HRNet Output"):
                #     HRNet_img_e.render()
                with gr.Tab("Grounding Dino"):
                    gdSAM_img_e.render()
                    segm_img_e = gr.JSON(visible=False)
        with gr.Row():
            threshold = gr.Slider(
                0.01,
                1,
                value=0.3,
                step=0.01,
                label="Threshold",
                interactive=True,
                info="Choose between 0.01 and 1",
                visible=False,
            )
        with gr.Row():
            with gr.Column():
                prev_btn_e = gr.Button("prev", variant="secondary")
            with gr.Column():
                next_btn_e = gr.Button("next", variant="secondary")
        with gr.Row(variant="panel"):
            label_checkbox = gr.CheckboxGroup(
                choices=[],
                label="select label in present image",
                interactive=True,
                visible=False,
            )

        with gr.Row():
            with gr.Column():
                dropdown = gr.Dropdown()
            with gr.Column():
                add_btn_e = gr.Button("add", variant="secondary")
            with gr.Column():
                delete_btn_e = gr.Button("delete", variant="secondary")

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
    drive_data.change(
        fn=lambda value: (
            gr.update(
                value=hrnet_label[int(value == "drive dataset")],
                visible=(value != "drive dataset"),
            ),
            gr.update(visible=(value != "drive dataset")),
        ),
        inputs=drive_data,
        outputs=[label_list, threshold],
    )
    set_label_btn_e.click(
        fn=lambda value: (
            label_checkbox.update(choices=value.replace(", ", ",").split(",")),
            label_checkbox.update(choices=value.replace(", ", ",").split(",")),
        ),
        inputs=label_list,
        outputs=[label_checkbox, dropdown],
    )
    # label_checkbox.change(
    #     fn=lambda value: gr.update(choices=value.replace(", ", ",").split(",")),
    #     inputs=label_checkbox,
    #     outputs=dropdown,
    # )
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
    add_btn_e.click(
        fn=lambda coord, segm, dropdown, cond_img: gr.update(
            value=add_mask(coord, segm, dropdown, cond_img), color_map=color_dict
        ),
        inputs=[coord, segm_img_e, dropdown, cond_img_e],
        outputs=gdSAM_img_e,
    )
    delete_btn_e.click(
        fn=lambda coord, segm, dropdown, gdsam: gr.update(
            value=delete_mask(coord, segm, dropdown, gdsam), color_map=color_dict
        ),
        inputs=[coord, segm_img_e, dropdown, gdSAM_img_e],
        outputs=gdSAM_img_e,
    )
    finish_btn_e.click(finish, inputs=id)
    save_btn_e.click(save_annotation, inputs=id)
    ################################################

    cond_img_e.select(get_points, inputs=cond_img_e, outputs=coord)

    present_img.change(
        fn=viz_img,
        inputs=[id, present_img],
        outputs=[cond_img_e],
    )
demo.queue()
demo.launch()
