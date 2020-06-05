import config

import pytorch_lightning as pl
import torch
from dataset import WheatDataset, WheatTest, AwgDataset, EffDataset
import config

import numpy as np
import pandas as pd

from utils import collate_fn, bboxtoIoU, format_prediction_string, bboxtoIP

class EffWheat(pl.LightningModule):
    
    def __init__(self, hparams, train_folds,  valid_folds, model = None):
        super(EffWheat, self).__init__()
        self.hparams = hparams
        self.model = model
        self.train_folds = train_folds
        self.valid_folds = valid_folds

        self.test_df = pd.DataFrame(columns=['image_id', 'PredictionString'])

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def train_dataloader(self):


        train_df = pd.read_csv(config.TRAIN_FOLDS)
        train_df = train_df[train_df.kfold.isin(self.train_folds)].reset_index(drop=True)

        data = AwgDataset(train_df, config.TRAIN_PATH)


        train_loader = torch.utils.data.DataLoader(dataset=data,
                                                   batch_size=config.TRAIN_BATCH_SIZE,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
        return train_loader

    def val_dataloader(self):

        valid_df = pd.read_csv(config.TRAIN_FOLDS)
        valid_df = valid_df[valid_df.kfold.isin(self.valid_folds)].reset_index(drop=True)

        data = AwgDataset(valid_df, config.TRAIN_PATH)

        valid_loader = torch.utils.data.DataLoader(dataset=data,
                                                   batch_size=config.VAL_BATCH_SIZE,
                                                   shuffle=False,
                                                   collate_fn=collate_fn)

        return valid_loader


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, mode='min', patience=7)
        warm_restart = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=674)

        return [optimizer], [scheduler, warm_restart]

    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        # boxes = targets['boxes']
        # labels = targets['labels']

        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_dict['net_loss'] = losses

        return {'loss': losses, 'log': loss_dict, 'progress_bar': loss_dict}

    def validation_step(self, batch, batch_idx):
        
        images, targets, _ = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        # print(targets)
        boxes = [t['boxes'] for t in targets]
        labels = [t['labels'] for t in targets]
        # print(boxes)
        # print(labels)

        # loss_dict = self.model(images, boxes, labels)
        loss_dict = self.model(images, targets)
        # AP = bboxtoIP(outputs, targets)
        # scores = bboxtoIoU(outputs, targets)
        print(loss_dict)
        return {'val_AP': AP, 'val_IoU': scores}

    def validation_epoch_end(self, outputs):
        
        metric = []
        IoU = []
        for o in outputs:
            metric.append(np.mean(o['val_AP']))
            IoU.append(np.mean(o['val_IoU']))
        
        metric = np.mean(metric)
        metric = torch.tensor(metric, dtype=torch.float)
        IoU = np.mean(IoU)
        IoU = torch.tensor(IoU, dtype=torch.float)

        tensorboard_logs = {'val_loss': -IoU, 'val_acc': IoU, 'val_IoU': IoU, 'val_P': metric}
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