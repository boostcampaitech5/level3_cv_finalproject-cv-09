import torch
from models.clip_seg import CLIPDensePredT
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_data
def load_image(image_path):
    return Image.open(image_path)

def predict(selected_image, prompts):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])
    img = transform(selected_image).unsqueeze(0)
    
    with torch.no_grad():
        repeat_num = len(prompts)
        preds = model(img.repeat(repeat_num,1,1,1).cuda(), prompts)[0]
    print("pred shape : ", preds.shape)
    
    return preds

def visualize_heatmap(preds, prompts):
    # visualize prediction
    fig, ax = plt.subplots(1, len(prompts) + 1, figsize=(3*(len(prompts) + 1), 4))
    [a.axis('off') for a in ax.flatten()]
    ax[0].imshow(selected_image)
    [ax[i+1].imshow(torch.sigmoid(preds[i].squeeze())) for i in range(len(prompts))];
    [ax[i+1].text(0, -15, prompt) for i, prompt in enumerate(prompts)];
    
    st.pyplot(fig)
    
def visualize_segmentation(preds, threshold):
    flat_preds = torch.sigmoid(preds.squeeze()).reshape((preds.shape[0], -1))

    # Initialize a dummy "unlabeled" mask with the threshold
    flat_preds_with_treshold = torch.full((preds.shape[0] + 1, flat_preds.shape[-1]), threshold)
    flat_preds_with_treshold[1:preds.shape[0]+1,:] = flat_preds

    # Get the top mask index for each pixel
    inds = torch.topk(flat_preds_with_treshold, 1, dim=0).indices.reshape((preds.shape[-2], preds.shape[-1]))

    segmentation_figure = plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(inds)
    st.pyplot(segmentation_figure)

if __name__ == "__main__":
    st.title('CLIP-seg Demo')

    # load model
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64).cuda()
    model.eval();

    # non-strict, because we only stored decoder weights (not CLIP weights)
    model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False);

    # load and normalize image
    # input_image = Image.open('images/N-B-P-004_000433.jpg')

    # or load from URL...
    # image_url = 'https://farm5.staticflickr.com/4141/4856248695_03475782dc_z.jpg'
    # input_image = Image.open(requests.get(image_url, stream=True).raw)

    images = ['images/N-B-P-004_000433.jpg', 
            'images/N-B-P-004_017137.jpg', 
            'images/N-B-P-021_000109.jpg',
            'images/N-E-C-020_000505.jpg',
            'images/N-E-C-020_002305.jpg',
            'images/N-E-H-010_000469.jpg',
            'images/S-W-P-004_015841.jpg']  # 이미지 파일 리스트
    # 이미지를 병렬로 출력
    columns = st.columns(len(images) // 2)
    for i, column in enumerate(columns):
        image = load_image(images[i])
        image2 = load_image(images[i + (len(images) // 2)])
        column.image(image, use_column_width=True, caption=f'Image {i+1}')
        column.image(image2, use_column_width=True, caption=f"Image{i + (len(images) // 2) + 1}")

    # 이미지 선택
    selected_index = st.radio('Select an image:', list(range(len(images))))

    # 선택된 이미지 출력
    selected_image = load_image(images[selected_index])
    st.image(selected_image, use_column_width=True, caption='Selected Image')

    if "text_inputs" not in st.session_state:
        st.session_state.text_inputs = []
       
    if "prev_prompt" not in st.session_state: 
        st.session_state.prev_prompt = ""
        
    user_prompt = st.text_input(label="Enter prompt", value="", key=f"input_{i+1}")
    if st.button('Enter') or (st.session_state.prev_prompt != user_prompt):
        st.session_state.prev_prompt = user_prompt
        st.session_state.text_inputs.append(st.session_state.prev_prompt)
        
    # 입력받은 텍스트 출력
    st.write("Entered Texts:")
        
    # 텍스트 입력란과 삭제 버튼을 생성하고 사용자로부터 값을 입력받음
    for i in range(len(st.session_state.text_inputs)):
        col1, col2 = st.columns([8, 2])
        with col1:
            st.write(st.session_state.text_inputs[i])
        with col2:
            delete_button = st.button(f"Delete {i+1}")
            if delete_button:
                st.session_state.text_inputs.pop(i)
                st.experimental_rerun()
    
    threshold = 0.5
    threshold_slider = st.slider("Threshold", min_value=0.0, max_value=1.0, value=threshold, step=0.01)
    
    # predict
    if st.button("predict"):
        # predict_and_visualize(selected_image, st.session_state.text_inputs)
        preds = predict(selected_image, ['person', 'tree', 'sky', 'shadow']).cpu()
        visualize_heatmap(preds, ['person', 'tree', 'sky', 'shadow'])
        visualize_segmentation(preds, threshold)