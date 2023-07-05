import streamlit as st
import numpy as np
from PIL import Image
import torch 
import cv2 
import json
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import random 
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
st.set_page_config(initial_sidebar_state="expanded")
st.title("Data Annotation Demo")
st.subheader("CV-09 Team")

colors = [(0, 0, 0), (0.8196078431372549, 0.2901960784313726, 0.25882352941176473), (0.42745098039215684, 0.9490196078431372, 0.2), (0.9490196078431372, 0.9254901960784314, 0.8862745098039215), (0.5764705882352941, 0.19607843137254902, 0.6235294117647059), (0.0196078431372549, 0.41568627450980394, 0.9725490196078431), (0.3764705882352941, 0.20784313725490197, 0.09411764705882353), (0.12156862745098039, 0.4745098039215686, 0.38823529411764707), (0.00392156862745098, 0.34901960784313724, 0.01568627450980392), (0.4470588235294118, 0.00392156862745098, 0.03137254901960784), (0.32941176470588235, 0.34901960784313724, 0.7607843137254902)]

with st.sidebar:
    mode = st.sidebar.radio(
        "Select platform",
        ("COCODataset", "etc")
    )

def main():

    entire_label_list =  st.text_input('Input entire label list', '(ex: human,dog,cat)')
    entire_label_list = entire_label_list.replace(', ', ',').split(',')

    label_list = st.multiselect(
        'Input label list about Current Image',
        entire_label_list
    )
    threshold = st.slider('Define threshold.', 0.0, 1.0, 0.5)
    uploaded_img = st.file_uploader("Input your unlabeled Image.")
    if uploaded_img is not None:
        image = Image.open(uploaded_img)
        inputs = processor(text=label_list, images=[image] * len(label_list), padding="max_length", return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)

        preds = outputs.logits.unsqueeze(1).cpu()
        flat_preds = torch.sigmoid(preds.squeeze()).reshape((preds.shape[0], -1))
        flat_preds_with_treshold = torch.full((preds.shape[0] + 1, flat_preds.shape[-1]), threshold)
        flat_preds_with_treshold[1:preds.shape[0]+1,:] = flat_preds
        inds = torch.topk(flat_preds_with_treshold, 1, dim=0).indices #.reshape((preds.shape[-2], preds.shape[-1])).unsqueeze(0)
        temp_list = []
        for i in inds.squeeze():
            temp_list.append(colors[i])

        image = cv2.resize(np.array(image), (preds.shape[-2], preds.shape[-1]))
        
        output = np.array(temp_list).T.reshape(3, preds.shape[-2], preds.shape[-1]).transpose(1,2,0) * 255

        blended = cv2.addWeighted(image, 0.5, output, 0.5, 0, dtype=cv2.CV_8UC3)
        #st.image(image)
        #st.image(output, clamp=True)
        st.image(blended, clamp=True)
        
if __name__=="__main__":
    main()