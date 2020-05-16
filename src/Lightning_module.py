import pytorch_lightning as pl
import torch
from dataset import WheatDataset
import config

import numpy as np

class LitWheat(pl.LightningModule):
    
    def __init__(self, train_folds,  valid_folds, model = None):
        super(LitWheat, self).__init__()
        # self.hparams = hparams
        self.model = model
        self.train_folds = train_folds
        self.valid_folds = valid_folds

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(WheatDataset(folds=self.train_folds),
                                                   batch_size=config.TRAIN_BATCH_SIZE,
                                                   shuffle=True,
                                                   collate_fn=self.collate_fn)
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(WheatDataset(folds=self.valid_folds),
                                                   batch_size=config.VAL_BATCH_SIZE,
                                                   shuffle=False,
                                                   collate_fn=self.collate_fn)

        return valid_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, mode='min', patience=2)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        return {'loss': losses, 'log': loss_dict, 'progress_bar': loss_dict}

    def validation_step(self, batch, batch_idx):
        
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images)
        scores = self.bboxtoIoU(outputs, targets)
        return {'val_IoU': scores}

    def validation_epoch_end(self, outputs):
        
        metric = []
        for o in outputs:
            metric.append(np.mean(o['val_IoU']))
        metric = np.mean(metric)
        tensorboard_logs = {'val_loss': -metric, 'val_acc': metric}
        return {'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def bboxtoIoU(self, results, targets):
        IoU = []
        for i, (res, gt) in enumerate(zip(results, targets)):
            b_res = res['boxes'].cpu().detach().numpy()
            b_gt = gt['boxes'].cpu().detach().numpy()

            score = self.run(b_res, b_gt)
            mean = np.mean(np.max(score, axis=0))
            IoU.append(mean)
        return IoU
            
    def run(self, bboxes1, bboxes2):
        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea + 1e-2)
        return iou