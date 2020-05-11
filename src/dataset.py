import albumentations as A
import numpy as np
import pandas as pd
# import cv2
from PIL import Image
import torch

import config

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
            ], bbox_params={'format':'coco', 'min_area': 1, 'min_visibility': 0.5, 'label_fields': ['labels']})
        else:
            self.aug = A.Compose([
                A.Resize(config.CROP_SIZE, config.CROP_SIZE, always_apply=True),
                A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.4),
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
                    A.Blur(),
                    A.NoOp()
                ]),
                A.Normalize(config.MODEL_MEAN, config.MODEL_STD, always_apply=True)
            ], bbox_params={'format':'coco', 'min_area': 1, 'min_visibility': 0.5, 'label_fields': ['labels']}, p=1.0)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        img_name = self.image_ids[item]
        image = np.array(Image.open(f'{config.TRAIN_PATH}/{img_name}.jpg'))
        bboxes_id = self.df[self.df.image_id == img_name].index.tolist()
        # print(len(bboxes))
        bboxes = self.df.loc[bboxes_id, ['x','y','w','h']].values
        # bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        # bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

        num_bboxes = len(bboxes)
        cat_id = [1.0]*num_bboxes

        # print(bboxes)
        # print(cat_id)

        # image': image, 'bboxes': bboxes
        # sample = {
        #         'image': image,
        #         'bboxes': bboxes,
        #         'labels': cat_id
        #     }


        category_id_to_name = {1: 'wheat'}

        # augmented = aug(**sample)

        augmented = self.aug(image=image, bboxes=bboxes, labels=cat_id)

        image = augmented['image'].copy()
        bboxes = augmented['bboxes'].copy()

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        

        bboxes = torch.tensor(bboxes, dtype=torch.float)
        labels = torch.ones((num_bboxes,), dtype=torch.int64)

        target = {}
        target['bboxes'] = bboxes
        target['labels'] = labels

        return {
            'image': torch.tensor(image, dtype=torch.float),
            'target': target
        }



