import os
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import torch
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from utils.tools_gradio import fast_process

from PIL import Image
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
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
        use_retina=use_retina
    )
    return fig

def get_points(image, evt: gr.SelectData):
    x, y = evt.index[0], evt.index[1]
    pixels = image.load()
    return x, y, pixels[x, y]


cond_img_e = gr.Image(label="Input", value=default_example[0], type="pil")
segm_img_e = gr.Image(label="Segmented Image", interactive=False, type="pil")

modified_img_e = gr.Image(label="Modified Image", interactive=True, type="pil", image_mode='RGBA')
mask_img_e = gr.Image(label="mask Image", interactive=False, type="pil", image_mode='RGBA')

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

    with gr.Tab("Everything mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_e.render()
            with gr.Column(scale=1):
                input_size_slider.render()
                segment_btn_e = gr.Button(
                    "Segment Everything", variant="primary"
                )
                with gr.Accordion("Advanced options", open=True):
                    mor_check = gr.Checkbox(
                        value=True,
                        label="better_visual_quality",
                        info="better quality using morphologyEx",
                    )

                    retina_check = gr.Checkbox(
                        value=True,
                        label="use_retina",
                        info="draw high-resolution segmentation masks",
                    )
        # Submit & Clear
        with gr.Row():
            with gr.Column(scale=1):
                segm_img_e.render()
                with gr.Row():        
                    add_btn_e = gr.Button("add", variant="secondary")
                    delete_btn_e = gr.Button("delete", variant="secondary")
    
                # gr.Markdown("Try some of the examples below ⬇️")
                # gr.Examples(
                #     examples=examples,
                #     inputs=[cond_img_e],
                #     outputs=segm_img_e,
                #     fn=segment_everything,
                #     cache_examples=True,
                #     examples_per_page=4,
                # )
    
            with gr.Column():
                modified_img_e.render()
        with gr.Row():
            coord_value = gr.Textbox()
    #with gr.Row():
    #    with gr.Column(scale=1):
    #        mask_img_e.render()
    #    with gr.Column(scale=1):
    #        t_box = gr.Textbox(interactive=False)
            
    segment_btn_e.click(
        segment_everything,
        inputs=[
            cond_img_e,
            input_size_slider,
            mor_check,
            retina_check,
        ],
        outputs=[segm_img_e],
    )

    def clear():
        return None, None

    def clear_text():
        return None, None, None

    #clear_btn_e.click(clear, outputs=[cond_img_e, segm_img_e])
    segm_img_e.select(get_points, inputs=[segm_img_e], outputs=[coord_value])
demo.queue()
demo.launch()

'''
이미지를 원본이랑 마스크로 분리, 
modified_img_e 이미지 select 될때 마우스 좌표를 
마스크 이미지에 적용, 해당 픽셀의 값 반환 

'''