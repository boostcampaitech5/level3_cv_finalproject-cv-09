from PIL import Image
from collections import namedtuple
import torch
import numpy as np
import os

class CustomCityscapesSegmentation(torch.utils.data.Dataset):
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    cmap = []
    label = []
    for i in classes:
        if i.train_id >=0 and i.train_id <19:
            cmap.append(i.color)
            label.append(i.name)

    def __init__(self, data_dir, image_set="train", transform=None, target_transform=None):
        self._ignore_index = [255]
        
        self.data_dir = data_dir
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        self.path_jpeg = os.path.join(data_dir,'leftImg8bit',image_set)
        self.path_mask = os.path.join(data_dir,'gtFine',image_set)
        self.images = []
        self.targets = []

        for city in os.listdir(self.path_jpeg):
            img_dir = os.path.join(self.path_jpeg,city)
            target_dir = os.path.join(self.path_mask,city)
            for file_name in os.listdir(img_dir):
                target_name ='{}_{}'.format(file_name.split('_leftImg8bit')[0],'gtFine_labelIds.png')
                self.images.append(os.path.join(img_dir,file_name))
                self.targets.append(os.path.join(target_dir,target_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])

        image = np.array(image,dtype=np.uint8)
        target = np.array(target,dtype=np.uint8)
        
        for l in self.classes:
            idx = target==l.id
            target[idx] = l.train_id

        image = Image.fromarray(image)
        target = Image.fromarray(target)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.transform(target)
            
        return image, target
    
    
class CustomKRLoadSegmentation(torch.utils.data.Dataset):
    KRLoadClass = namedtuple('KRLoadClass', ['name', 'id', 'color'])

    classes = [
        KRLoadClass('background', 0, (0,0,0)),      # 배경
        KRLoadClass('wheelchair', 1, (255, 0, 0)),  # 휠체어
        KRLoadClass('carrier', 2, (0, 64, 0)),     # 화물차
        KRLoadClass('stop', 3, (0 ,255, 255)),     # 정지선
        KRLoadClass('cat', 4, (64, 0, 0)),         # 고양이
        KRLoadClass('pole', 5, (0, 128, 128)),     # 대
        KRLoadClass('traffic_light',6, (255, 0, 255)),  # 신호등
        KRLoadClass('traffic_sign', 7, (0, 0, 255)),    # 교통 표지판
        KRLoadClass('stroller', 8, (255, 255, 0 )), # 유모차
        KRLoadClass('dog', 9, (255, 128, 255)),    # 개
        KRLoadClass('barricade', 10, (0, 192, 0)),  # 바리케이드
        KRLoadClass('person', 11, (128, 0, 128)),    # 사람 
        KRLoadClass('scooter', 12, (128, 128, 0)),  # 스쿠터
        KRLoadClass('car', 13, (0, 0, 64)),         # 차
        KRLoadClass('truck', 14, (0, 255, 0)),       # 트럭
        KRLoadClass('bus', 15, (64, 64, 0)),        # 버스 
        KRLoadClass('bollard', 16, (64, 0, 64)),    # 인도 블럭 바리케이드 비슷한거
        KRLoadClass('motorcycle', 17, (128, 0, 255)),   # 오토바이
        KRLoadClass('bicycle', 18, (0, 64, 64)),    # 자전거
    ]
    class_names = [i.name for i in classes]

    cmap = []
    for i in classes:
        if i.id >=0 and i.id <19:
            cmap.append(i.color)

    def __init__(self, data_dir, image_set="train", transform=None, target_transform=None):
        self._ignore_index = [255]
        
        self.data_dir = data_dir
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        self.path_png = os.path.join(data_dir,'imgs',image_set)
        self.path_mask = os.path.join(data_dir,'labels',image_set)
        self.images = []
        self.targets = []

        for folders in os.listdir(self.path_png):
            img_dir = os.path.join(self.path_png,folders)
            target_dir = os.path.join(self.path_mask,folders)
            for file_name in os.listdir(img_dir):
                if os.path.exists(os.path.join(target_dir,file_name)):
                    self.images.append(os.path.join(img_dir,file_name))
                    self.targets.append(os.path.join(target_dir,file_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])

        image = np.array(image,dtype=np.uint8)
        target = np.array(target,dtype=np.uint8)

        image = Image.fromarray(image)
        target = Image.fromarray(target)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.transform(target)
            
        return image, target