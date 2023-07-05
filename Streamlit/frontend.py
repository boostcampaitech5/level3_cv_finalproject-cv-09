# This is file to implement brief sample about connection between Streamlit and FastAPI.

import streamlit as st
import requests

st.title("Sample Connection")

image = st.file_uploader("Choose an image")
upload_id = st.text_input(label='Enter ID to upload')

if st.button('Upload Image'):
    if image is not None and upload_id is not None:
        files = {'file' : image}
        res = requests.post(f"http://localhost:30008/upload/{int(upload_id)}", files=files)
        st.write(res.content)


model_id = st.text_input(label='Enter input image ID')
prompts = st.text_input(label='Enter prompts')
if st.button('Run Model'):
    if model_id is not None and prompts is not None:
        res=requests.post(f"http://localhost:30008/predict/{int(model_id)}_{prompts}")
        if res.status_code == 200:
            st.write('Done!')

download_id = st.text_input(label='Enter ID to download')
if st.button('Show Result'):
    if download_id is not None:
        res = requests.get(f"http://localhost:30008/download/{int(download_id)}")
        if res.status_code == 200:
            st.image(res.content)
            st.download_button(label='Download', data=res.content, file_name=f'{download_id}.jpg')

log_id = st.text_input(label='Enter ID to feedback')
log = st.text_input(label='Enter feedback')
if st.button('Send Feedback'):
    if log is not None:
        data = {'log' : log}
        res = requests.post(f"http://localhost:30008/log/{int(log_id)}", json=data)