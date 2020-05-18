import albumentations as A
import numpy as np
import pandas as pd
# import cv2
from PIL import Image
import torch

import config

from albumentations.pytorch import ToTensorV2

import os

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
                    A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.4),
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
                    A.CLAHE(),
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

