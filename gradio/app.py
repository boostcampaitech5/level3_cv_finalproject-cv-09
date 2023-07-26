import os
import gradio as gr
import numpy as np
import shutil
import torch
import json
import requests
from PIL import Image
from zipfile import ZipFile
from collections import deque
from torchvision.utils import draw_segmentation_masks


save_annotate = []
hrnet_label = [
    "",
    "background,traffic_light_controller,wheelchair,truck,traffic_sign,traffic_light,stroller,stop,scooter,pole,person,motorcycle,dog,cat,carrier,car,bus,bollard,bicycle,barricade",
]
classes = {
    "traffic_light_controller": [0, (0, 0, 255)],
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
    "background": [255, (0, 0, 0)],  # 배경
}
color_dict = {
    "background": "#ffffff",
    "traffic_light_controller": "#0000ff",
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
global_points = []
global_point_label = []


def draw_image(image, masks, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if len(masks) > 0:
        image = draw_segmentation_masks(
            image, masks=masks, colors=list(color_dict.values()), alpha=alpha
        )
    return Image.fromarray(image.numpy().transpose(1, 2, 0))


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


def rle_encode(mask):
    mask_flatten = mask.flatten()
    mask_flatten = np.concatenate([[0], mask_flatten, [0]])
    runs = np.where(mask_flatten[1:] != mask_flatten[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = " ".join(str(x) for x in runs)
    return rle


def zip_upload(is_drive, img_zip, id):
    with ZipFile(img_zip.name, "r") as f:
        f.extractall(f"data/{id}")
    data = {"id": str(id)}
    with open(img_zip.name, "rb") as f:
        files = {"files": f}
        res = requests.post(
            "http://127.0.0.1:30008/zip_upload/",
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
    seg = requests.post("http://127.0.0.1:30008/segment/", data=data)
    mask_dict = json.loads(seg.content)

    return mask_dict


def hrnet_request(id, img_path):
    img_prefix = f"data/{id}"
    image_pil = Image.open(os.path.join(img_prefix, img_path)).convert("RGB")
    data = {"path": os.path.join(str(id), str(img_path))}
    res = requests.post("http://127.0.0.1:30008/segment_hrnet/", data=data)
    mask_dict = json.loads(res.content)

    temp = []
    for label, mask in mask_dict["masks"].items():
        seg_mask = rle_decode(mask, mask_dict["size"])
        temp.append(seg_mask)
    global save_annotate
    save_annotate = temp
    return draw_image(np.array(image_pil), torch.tensor(np.array(temp), dtype=bool))


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
    seg = requests.post("http://127.0.0.1:30008/segment_text/", data=data)
    mask_dict = json.loads(seg.content)

    temp = []
    # asyncio.run(make_annotation_json(img_path, mask_dict))
    for label, mask in mask_dict["masks"].items():
        seg_mask = rle_decode(mask, mask_dict["size"])
        temp.append((seg_mask, label))

    return image_pil, temp


def segment_request(id, threshold, img_path, text_prompt):
    return hrnet_request(id, img_path)


def json_upload(id):
    with ZipFile(f"data/{id}/annotations.zip", "r") as f:
        f.extractall(f"data/{id}")
    data = {"id": str(id)}
    with open(f"data/{id}/annotations.zip", "rb") as f:
        files = {"files": f}
        res = requests.post(
            "http://127.0.0.1:30008/json_upload/",
            data=data,
            files=files,
        )


def finish(id):
    data = {"id": str(id)}
    shutil.make_archive(f"data/{id}/annotation", "zip", f"data/annotations/{id}")

    with open(f"data/{id}/annotation.zip", "rb") as f:
        files = {"files": f}
        upload_res = requests.post(
            "http://127.0.0.1:30008/json_upload/", data=data, files=files
        )

    remove_res = requests.post("http://127.0.0.1:30008/remove/", data=data)
    return f"data/{id}/annotation.zip"


def save_annotation(id, img_path, image):
    w, h = image.size
    anno_dict = {}
    anno_dict["size"] = [w, h]
    anno_dict["image_path"] = img_path
    anno_dict["masks"] = []
    img_path = img_path.split(".")[0] + ".json"
    save_path = f"data/annotations/{id}"
    global save_annotate

    for mask, label in zip(save_annotate, list(classes.keys())):
        anno_dict["masks"].append({"label": label, "mask": rle_encode(mask)})
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, img_path), "w") as f:
        json.dump(anno_dict, f, indent=4)
    return "\n".join(os.listdir(f"data/annotations/{id}"))


def clear(coord):
    coord = []
    return coord


def modify(id, img_path, label, image, coord):
    points = [[i[0], i[1]] for i in coord]
    labels = [i[2] for i in coord]
    data = {
        "path": os.path.join(str(id), str(img_path)),
        "global_points": points,
        "global_point_label": labels,
    }
    res = requests.post("http://127.0.0.1:30008/segment/", data=data)
    mask_array = json.loads(res.content)

    temp = modify_label(np.array(mask_array), label)
    return draw_image(np.array(image), torch.tensor(np.array(temp), dtype=bool))


def modify_label(mask, label):
    global save_annotate
    mask = Image.fromarray(mask).resize(
        (save_annotate[0].shape[1], save_annotate[0].shape[0])
    )
    mask.save("mask.png")
    mask = np.array(mask)
    temp = []
    for i, m in enumerate(save_annotate):
        if i == classes[label][0]:
            temp.append(np.logical_or(m, mask))
        else:
            temp.append(np.logical_and(m, np.logical_not(mask)))

    save_annotate = temp
    return temp


# Description
title = "<center><strong><font size='8'>Image Annotation Tool<font></strong></center>"

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


def add_points(image, add_del, coord, evt: gr.SelectData):
    x, y = evt.index[0], evt.index[1]
    coord.append([x, y, True if add_del == "add label" else False])
    return coord


origin_img_e = gr.Image(label="Original", interactive=False, type="pil")
cond_img_e = gr.Image(label="Input", interactive=False, type="pil")


# temp = [HRNet_img_e, gdSAM_img_e]
id = gr.Textbox()
img_list = gr.JSON()
drive_data = gr.Radio(["walking dataset", "others"])


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
            with gr.Column(scale=3):
                present_img = gr.Textbox(label="present Image name", interactive=False)
            with gr.Column(scale=1):
                request_btn_e = gr.Button("request", variant="primary")
        with gr.Row():
            with gr.Tab("Original Image"):
                origin_img_e.render()
                segm_img_e = gr.JSON(visible=False)
            with gr.Tab("Table Image"):
                cond_img_e.render()
                coord = gr.JSON([])
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
                clear_btn_e = gr.Button("clear")
            with gr.Column():
                dropdown = gr.Dropdown(interactive=True)
            with gr.Column():
                add_del_radio = gr.Radio(["add label", "delete label"])
            with gr.Column():
                modify_btn_e = gr.Button("modify")
        with gr.Row():
            with gr.Column():
                prev_btn_e = gr.Button("prev", variant="secondary")
            with gr.Column():
                save_btn_e = gr.Button("save", variant="secondary")
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
            prev_annotation = gr.Textbox(label="Preview annotation.zip")
        with gr.Row():
            with gr.Column(scale=1):
                finish_btn_e = gr.Button("Finish")
            with gr.Column(scale=2):
                file_response = gr.File()

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
            gr.update(choices=value.replace(", ", ",").split(",")),
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
        outputs=[cond_img_e],
    )

    save_btn_e.click(
        save_annotation, inputs=[id, present_img, origin_img_e], outputs=prev_annotation
    )
    finish_btn_e.click(finish, inputs=id, outputs=file_response)
    clear_btn_e.click(clear, inputs=coord, outputs=coord)
    modify_btn_e.click(
        fn=modify,
        inputs=[id, present_img, dropdown, origin_img_e, coord],
        outputs=[cond_img_e],
    )
    ################################################

    cond_img_e.select(
        add_points, inputs=[cond_img_e, add_del_radio, coord], outputs=[coord]
    )

    present_img.change(
        fn=viz_img,
        inputs=[id, present_img],
        outputs=[origin_img_e],
    )
demo.queue()
demo.launch()
