import pytorch_lightning as pl
from Lightning_module import LitWheat
from model_dispatcher import MODEL_DISPATCHER
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import config

import os

from argparse import ArgumentParser


def train_iterative(train_folds,  valid_folds):

    parser = ArgumentParser()
    parser.add_argument('--lr', type=int, default=config.LR)
    hparams = parser.parse_args()

    model = MODEL_DISPATCHER[config.MODEL_NAME]

    model.load_state_dict(torch.load(config.MODEL_SAVE))

    model = freeze(model, 4)

    lit_model = LitWheat(hparams, train_folds=train_folds,  valid_folds=valid_folds, model=model)

    early_stopping = pl.callbacks.EarlyStopping(mode='min', monitor='val_loss', patience=10)
    # model_checkpoint = pl.callbacks.ModelCheckpoint(filepath= config.FILESAVE, mode='min', monitor='val_loss', verbose=True)

    trainer = pl.Trainer(
        gpus=1,
        accumulate_grad_batches=32,
        profiler=True,
        early_stop_callback=early_stopping,
        gradient_clip_val=0.5,
        debug=False,
        metric='val_loss'
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