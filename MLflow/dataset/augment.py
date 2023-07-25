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

