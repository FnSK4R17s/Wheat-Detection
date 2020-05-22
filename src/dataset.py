import albumentations as A
import numpy as np
import pandas as pd
# import cv2
from PIL import Image
import torch

import config

from albumentations.pytorch import ToTensorV2

import os

import random
from sklearn.utils import shuffle
from awsome_augs import load_mosaic, random_affine, augment_hsv, letterbox, load_image
from torch.utils.data import DataLoader, Dataset

class WheatDataset:
    def __init__(self, folds):
        df = pd.read_csv(config.TRAIN_FOLDS)
        df = df[['image_id','x','y','w','h','kfold']]
        self.df = df[df.kfold.isin(folds)].reset_index(drop=True)
        self.image_ids = self.df.image_id.unique()

        if len(folds) == 1:
            self.aug = A.Compose([
                A.Resize(config.CROP_SIZE, config.CROP_SIZE, always_apply=True),
                A.Normalize(config.MODEL_MEAN, config.MODEL_STD, always_apply=True)
            ], bbox_params={'format':config.DATA_FMT, 'min_area': 1, 'min_visibility': 0.5, 'label_fields': ['labels']})
        else:
            self.aug = A.Compose([
                A.Resize(config.CROP_SIZE, config.CROP_SIZE, always_apply=True),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
                    A.RandomGamma(gamma_limit=(50, 150)),
                    A.NoOp()
                ]),
                A.OneOf([
                    A.RGBShift(r_shift_limit=20, b_shift_limit=15, g_shift_limit=15),
                    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5),
                    A.NoOp()
                ]),
                A.OneOf([
                    A.ChannelShuffle(),
                    A.CLAHE(clip_limit=4),
                    A.NoOp()
                ]),
                A.OneOf([
                    A.JpegCompression(),
                    A.Blur(blur_limit=4),
                    A.NoOp()
                ]),
                A.Flip(),
                A.Normalize(config.MODEL_MEAN, config.MODEL_STD, always_apply=True)
            ], bbox_params={'format':config.DATA_FMT, 'min_area': 1, 'min_visibility': 0.5, 'label_fields': ['labels']}, p=1.0)

        # self.aug = A.Compose([
        #     A.Resize(config.CROP_SIZE, config.CROP_SIZE, always_apply=True),
        #     A.CLAHE(clip_limit=4),
        #     A.Normalize(config.MODEL_MEAN, config.MODEL_STD, always_apply=True)
        # ], bbox_params={'format':config.DATA_FMT, 'min_area': 1, 'min_visibility': 0.5, 'label_fields': ['labels']}, p=1.0)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        img_name = self.image_ids[item]
        image = np.array(Image.open(f'{config.TRAIN_PATH}/{img_name}.jpg'))
        bboxes_id = self.df[self.df.image_id == img_name].index.tolist()
        # print(len(bboxes))
        bboxes = self.df.loc[bboxes_id, ['x','y','w','h']].values
        if config.DATA_FMT == 'pascal_voc':
            bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
            bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

        num_bboxes = len(bboxes)
        cat_id = [1.0]*num_bboxes

        category_id_to_name = {1: 'wheat'}

        bboxes = torch.tensor(bboxes, dtype=torch.float)
        labels = torch.ones((num_bboxes,), dtype=torch.int64)

        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        iscrowd = torch.zeros((num_bboxes,), dtype=torch.int64)

        target = {}
        target['boxes'] = bboxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([item])
        target['area'] = area
        target['iscrowd'] = iscrowd

        sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }

        sample = self.aug(**sample)

        image = sample['image']

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        target['boxes'] = torch.as_tensor(sample['bboxes'], dtype=torch.float32)
        
        image = torch.as_tensor(image, dtype=torch.float32)

        return image, target, img_name



class WheatTest:
    def __init__(self):
        
        self.image_ids = []
        for file in os.listdir(config.TEST_PATH):
            self.image_ids.append(os.path.splitext(file)[0])

        self.aug = A.Compose([
            A.Resize(config.CROP_SIZE, config.CROP_SIZE, always_apply=True),
            A.Normalize(config.MODEL_MEAN, config.MODEL_STD, always_apply=True)
        ], p=1.0)
    
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        img_name = self.image_ids[item]
        image = np.array(Image.open(f'{config.TEST_PATH}/{img_name}.jpg'))

        target = {}
        target['boxes'] = torch.tensor([item])
        target['labels'] = torch.tensor([item])
        target['image_id'] = torch.tensor([item])
        target['area'] = torch.tensor([item])
        target['iscrowd'] = torch.tensor([item])

        sample = {
                'image': image,
            }

        sample = self.aug(**sample)

        image = sample['image']

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        image = torch.as_tensor(image, dtype=torch.float32)

        return image, target, img_name



class AwgDataset(Dataset):
    
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()
        
        self.df = dataframe
        self.image_ids = dataframe['image_id'].unique()
        self.image_ids = shuffle(self.image_ids)
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * len(self.image_ids)
        self.img_size = 1024
        im_w = 1024
        im_h = 1024
        for i, img_id in enumerate(self.image_ids):
            records = self.df[self.df['image_id'] == img_id]
            boxes = records[['x', 'y', 'w', 'h']].values
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            boxesyolo = []
            for box in boxes:
                x1, y1, x2, y2 = box
                xc, yc, w, h = 0.5*x1/im_w+0.5*x2/im_w, 0.5*y1/im_h+0.5*y2/im_h, abs(x2/im_w-x1/im_w), abs(y2/im_h-y1/im_h)
                boxesyolo.append([1, xc, yc, w, h])
            self.labels[i] = np.array(boxesyolo)
        
        self.image_dir = image_dir
        self.transforms = transforms
        
        self.mosaic = False
        self.augment = True

        self.aug = A.Compose([
            A.Resize(config.CROP_SIZE, config.CROP_SIZE, always_apply=True),
            A.Normalize(config.MODEL_MEAN, config.MODEL_STD, always_apply=True),
            ToTensorV2(p=1.0)
        ], p=1.0)

    def __getitem__(self, index: int):

        #img, labels = load_mosaic(self, index)
        self.mosaic = True
        if random.randint(0,1) ==0:
            self.mosaic = False
        if self.mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
        
        if self.augment:
            # Augment imagespace
            if not self.mosaic:
                img, labels = random_affine(img, labels,
                                            degrees=0,
                                            translate=0,
                                            scale=0,
                                            shear=0)

            # Augment colorspace
            augment_hsv(img, hgain=0.0138, sgain= 0.678, vgain=0.36)

        
        bboxes = []

        for _, a, b, c, d in labels:
            bboxes.append([a,b,c,d])

        bboxes = np.array(bboxes)
        img = np.array(img)

        num_bboxes = len(bboxes)
        iscrowd = torch.zeros((num_bboxes,), dtype=torch.int64)

        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.ones((num_bboxes,), dtype=torch.int64)

        target = {}
        target['boxes'] = bboxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd
        
        image = self.aug(image=img)['image']

        return image, target, 'awg'

    def __len__(self) -> int:
        return self.image_ids.shape[0]