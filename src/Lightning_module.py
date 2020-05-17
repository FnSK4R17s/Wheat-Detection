import pytorch_lightning as pl
import torch
from dataset import WheatDataset, WheatTest
import config

import numpy as np
import pandas as pd

from utils import collate_fn, bboxtoIoU, format_prediction_string

class LitWheat(pl.LightningModule):
    
    def __init__(self, train_folds,  valid_folds, model = None):
        super(LitWheat, self).__init__()
        # self.hparams = hparams
        self.model = model
        self.train_folds = train_folds
        self.valid_folds = valid_folds

        self.test_df = pd.DataFrame(columns=['image_id', 'PredictionString'])

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(WheatDataset(folds=self.train_folds),
                                                   batch_size=config.TRAIN_BATCH_SIZE,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(WheatDataset(folds=self.valid_folds),
                                                   batch_size=config.VAL_BATCH_SIZE,
                                                   shuffle=False,
                                                   collate_fn=collate_fn)

        return valid_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, mode='min', patience=4)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        return {'loss': losses, 'log': loss_dict, 'progress_bar': loss_dict}

    def validation_step(self, batch, batch_idx):
        
        images, targets, _ = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images)
        scores = bboxtoIoU(outputs, targets)
        return {'val_IoU': scores}

    def validation_epoch_end(self, outputs):
        
        metric = []
        for o in outputs:
            metric.append(np.mean(o['val_IoU']))
        metric = np.mean(metric)
        tensorboard_logs = {'val_loss': -metric, 'val_acc': metric}
        return {'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        images, _, img_name = batch
        
        results = []

        outputs = self.model(images)

        for i, image in enumerate(images):

            boxes = outputs[i]['boxes'].data.cpu().numpy()
            scores = outputs[i]['scores'].data.cpu().numpy()
            
            boxes = boxes[scores >= config.detection_threshold].astype(np.int32)
            scores = scores[scores >= config.detection_threshold]
            image_id = img_name[i]
            
            result = {
                'image_id': image_id,
                'PredictionString': format_prediction_string(boxes, scores)
            }

            results.append(result)

        df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

        # print(df.head())

        self.test_df = self.test_df.append(df, ignore_index=True)

        return results

    def test_epoch_end(self, outputs):

        # print(outputs)
        # print(self.test_df.head())
        # self.test_df.to_csv(config.SUB_FILE, index=False)
        return {}

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(WheatTest(),
                                                   batch_size=config.TEST_BATCH_SIZE,
                                                   shuffle=False,
                                                   collate_fn=collate_fn)

        return test_loader