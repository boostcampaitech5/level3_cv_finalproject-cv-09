import os
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import tqdm
from io import BytesIO
import argparse
import torch
from models.light import PLModel
import torch.nn.functional as F
import cv2
from dataset import CustomKRLoadSegmentation
from torchvision.transforms import ToTensor, Normalize

# def encode_mask_to_rle(mask):
#     '''
#     mask: numpy array binary mask 
#     1 - mask 
#     0 - background
#     Returns encoded run length 
#     '''
#     pixels = mask.flatten()
#     pixels = np.concatenate([[0], pixels, [0]])
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
#     runs[1::2] -= runs[::2]
#     return ' '.join(str(x) for x in runs)


# def decode_rle_to_mask(rle, height, width):
#     s = rle.split()
#     starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
#     starts -= 1
#     ends = starts + lengths
#     img = np.zeros(height * width, dtype=np.uint8)
    
#     for lo, hi in zip(starts, ends):
#         img[lo:hi] = 1
    
#     return img.reshape(height, width)

def test(model, image, tuple):
    args = get_arg()
    model = model.cuda()
    model.eval()
    mask_list = []
    
    with torch.no_grad():
        n_class = args.num_classes
        image = image.cuda()    
        logits = model(image)
        
        # restore original size
        outputs = torch.sigmoid(logits)
        # outputs = (outputs > 0.1).detach().cpu().numpy()
        #for i in range(n_class):
        #    outputs[:,i] = (outputs[:,i]> 0.5)
        outputs = outputs.argmax(dim=1)
        outputs = outputs.detach().cpu().numpy()
        result = outputs == np.arange(logits.shape[1])[:, np.newaxis, np.newaxis]
        for i in range(n_class):
            mask_list.append([tuple.label[i], result[i]])
        # result = np.zeros_like(logits.detach().cpu().numpy(), dtype=bool)
        # result[:,outputs,:,:] = True
            
    return outputs, mask_list



def get_arg():
    parser = argparse.ArgumentParser(description="mlflow-pytorch test")
    parser.add_argument("--accelerator", choices=['cpu','gpu','auto'],default='gpu')
    parser.add_argument("--precision", choices=['32','16'],default='16')
    parser.add_argument("--regist_name", type=str, default="register_model")
    parser.add_argument("--bn_type", type=str, default= 'torchbn')
    parser.add_argument("--num_classes", type=int, default= 19)
    parser.add_argument("--backbone", type=str, default='hrnet48')
    parser.add_argument("--pretrained", type=str, default='/opt/ml/level3_cv_finalproject-cv-09/MLflow/checkpoint/best.pth')
    parser.add_argument("--experiment_name",type=str, default='mlflow_ex')
    
    args = parser.parse_args()
    return args



def process_image_and_get_masks(img):
    args = get_arg()

    # Load and preprocess the image
    convert_tensor = transforms.Compose([ToTensor(),Normalize((0.286,0.325,0.283),(0.186,0.190,0.187))])
    image = convert_tensor(img)
    image = image.unsqueeze(0)

    # Initialize the lighiting model
    model = PLModel(args=args)

    # Get masks using the 'test' function
    masks = test(model, image, CustomKRLoadSegmentation)
    return masks

def mask_color(mask,tuple):
    cmap = tuple.cmap
    label = tuple.label
    if isinstance(mask,np.ndarray):
        r_mask = np.zeros_like(mask,dtype=np.uint8)
        g_mask = np.zeros_like(mask,dtype=np.uint8)
        b_mask = np.zeros_like(mask,dtype=np.uint8)
        for k in range(len(cmap)):
            indice = mask==k
            r_mask[indice] = cmap[k][0]
            g_mask[indice] = cmap[k][1]
            b_mask[indice] = cmap[k][2]
        return np.stack([b_mask, g_mask, r_mask], axis=2)


if __name__=="__main__":
    img = Image.open("/opt/ml/level3_cv_finalproject-cv-09/MLflow/test.jpg")

    mask,mask_list = process_image_and_get_masks(img)
    print(mask_list)
    # print(f"result shpae: {result.shape}, {result[12][540][1100]}")

    # print(f"mask shape: {mask.shape}")

    # 이미지 저장
    out = np.squeeze(mask,axis=0)
    print(out.shape,out.dtype)

    out = mask_color(out,CustomKRLoadSegmentation)
    output_path = './result.jpg'
    cv2.imwrite(output_path, out)