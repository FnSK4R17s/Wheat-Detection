import pytorch_lightning as pl
import torch
from dataset import WheatDataset
import config

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, mode='min', patience=15)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        # separate losses
        loss_dict = self.model(images, targets)
        # print(loss_dict)
        # total loss
        losses = sum(loss for loss in loss_dict.values())

        return {'loss': losses, 'log': loss_dict, 'progress_bar': loss_dict}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        outputs = self.model(images)

        # print(outputs)
        bboxes = [o['boxes'] for o in outputs]
        # print(targets)
        target_boxes = [o['boxes'] for o in targets]
        # print(bboxes[0])
        # print(target_boxes[0])
        # losses = sum(loss for loss in outputs.values())
        val = {'val_loss': outputs, 'log': outputs, 'progress_bar': outputs}
        return {}

    def validation_epoch_end(self, outputs):
        metric = 0
        tensorboard_logs = {'main_score': metric}
        return {'val_loss': metric, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def bboxtoIoU(self, boxes, targets):
        pass