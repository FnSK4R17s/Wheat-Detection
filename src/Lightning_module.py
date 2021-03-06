import pytorch_lightning as pl
import torch
from dataset import WheatDataset, WheatTest, AwgDataset
import config

import numpy as np
import pandas as pd

from utils import collate_fn, bboxtoIoU, format_prediction_string, bboxtoIP

from model_dispatcher import MODEL_DISPATCHER

class LitWheat(pl.LightningModule):
    
    def __init__(self, hparams, train_folds,  valid_folds):
        super(LitWheat, self).__init__()
        self.hparams = hparams
        self.model = MODEL_DISPATCHER[self.hparams.model_name]
        self.train_folds = train_folds
        self.valid_folds = valid_folds

        self.test_df = pd.DataFrame(columns=['image_id', 'PredictionString'])

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def train_dataloader(self):

        if self.hparams.aws:

            train_df = pd.read_csv(config.TRAIN_FOLDS)
            train_df = train_df[train_df.kfold.isin(self.train_folds)].reset_index(drop=True)

            data = AwgDataset(train_df, config.TRAIN_PATH)
        else:
            data = WheatDataset(folds=self.train_folds)


        train_loader = torch.utils.data.DataLoader(dataset=data,
                                                   batch_size=self.hparams.batch_size,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(WheatDataset(folds=self.valid_folds),
                                                   batch_size=self.hparams.batch_size,
                                                   shuffle=False,
                                                   collate_fn=collate_fn)

        return valid_loader


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, mode='min', patience=7)
        warm_restart = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(674/(self.hparams.accumulate)))

        return [optimizer], [scheduler, warm_restart]

    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_dict['net_loss'] = losses

        return {'loss': losses, 'log': loss_dict, 'progress_bar': loss_dict}

    def validation_step(self, batch, batch_idx):
        
        images, targets, _ = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images)
        AP = bboxtoIP(outputs, targets)
        scores = bboxtoIoU(outputs, targets)
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

        return {}

    def test_epoch_end(self, outputs):

        # print(outputs)
        # print(self.test_df.head())
        # self.test_df.to_csv(config.SUB_FILE, index=False)
        return {}

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(WheatTest(),
                                                   batch_size=self.hparams.batch_size,
                                                   shuffle=False,
                                                   collate_fn=collate_fn)

        return test_loader