import pytorch_lightning as pl
from Lightning_module import LitWheat
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import config

import os

from argparse import ArgumentParser


def train_iterative(train_folds,  valid_folds):

    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=config.LR)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--model_name', type=str, default=config.MODEL_NAME)
    parser.add_argument('--accumulate', type=int, default=config.ACCUMULATE)
    parser.add_argument('--aws', type=bool, default=True)
    hparams = parser.parse_args()


    # model.load_state_dict(torch.load(config.MODEL_SAVE))

    # model = freeze(model, 4)

    lit_model = LitWheat(hparams, train_folds=train_folds,  valid_folds=valid_folds)

    early_stopping = pl.callbacks.EarlyStopping(mode='min', monitor='val_loss', patience=50)
    model_checkpoint = pl.callbacks.ModelCheckpoint(filepath=config.MODEL_SAVE, save_weights_only=False, mode='max', monitor='val_IoU', verbose=False)

    trainer = pl.Trainer(
        gpus=1,
        accumulate_grad_batches=hparams.accumulate,
        profiler=True,
        early_stop_callback=early_stopping,
        checkpoint_callback=model_checkpoint,
        gradient_clip_val=0.5,
        debug=False,
        metric='net_loss',
        auto_lr_find=True
    )

    trainer.fit(lit_model)

    torch.save(lit_model.model.state_dict(), config.MODEL_SAVE)

    trainer.test()

    lit_model.test_df.to_csv(config.SUB_FILE, index=False)
    print(lit_model.test_df.head())

def makepaths(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def freeze(model, level):

    for layer in model.backbone.parameters():
        layer.requires_grad = True 

    for layer in model.backbone.fpn.parameters():
        layer.requires_grad = True

    for layer in model.backbone.body.layer4.parameters():
        layer.requires_grad = True

    return model

if __name__ == "__main__":

    makepaths([config.MODEL_PATH, config.PATH, config.FILEPATH])

    train_iterative([0, 1, 2, 3],[4])