import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import numpy as np
import random

def get_train_transform(train=True,crop_size=(769,769)):
    if train:
        transform = A.Compose([
            A.RandomCrop(crop_size[0], crop_size[1]),
            A.Rotate(limit=30),
            A.HorizontalFlip(),
            A.Normalize(mean=(0.286,0.325,0.283),std=(0.186,0.190,0.187)), # using cityscape values
            ToTensorV2()
            ])
    else:
        transform = get_test_transform()
    
    return transform


def get_test_transform():
    transform = A.Compose([
            A.Normalize(mean=(0.286,0.325,0.283),std=(0.186,0.190,0.187)), # using cityscape values
            ToTensorV2()
            ])
    
    return transform


def cutmix_collate_fn(batch):
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int16(W * cut_rat)
        cut_h = np.int16(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

    p = random.random()
    img, label = zip(*batch)
    img = torch.stack(img,0)
    label = torch.stack(label,0)
    if p > 0.5:    
        indice = torch.randperm(len(batch))
        alpha = np.random.beta(1,1)
        
        shuffle_img = img[indice]
        shuffle_label = label[indice]
        
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(),alpha)
        img[:,:,bby1:bby2,bbx1:bbx2] = shuffle_img[:,:,bby1:bby2,bbx1:bbx2]
        label[:,bby1:bby2,bbx1:bbx2] = shuffle_label[:,bby1:bby2,bbx1:bbx2]

        return img,label
    else:
        return img,label