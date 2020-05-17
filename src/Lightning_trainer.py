import pytorch_lightning as pl
from Lightning_module import LitWheat
from model_dispatcher import MODEL_DISPATCHER
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import config

import os


def train_iterative(train_folds,  valid_folds):
    model = MODEL_DISPATCHER[config.MODEL_NAME]
    lit_model = LitWheat(train_folds=train_folds,  valid_folds=valid_folds, model=model)

    early_stopping = pl.callbacks.EarlyStopping(mode='min', monitor='val_loss', patience=10)
    # model_checkpoint = pl.callbacks.ModelCheckpoint(mode='max', monitor='main_score', verbose=True)

    trainer = pl.Trainer(
        gpus=1,
        accumulate_grad_batches=64,
        profiler=True,
        early_stop_callback=early_stopping,
        gradient_clip_val=0.5,
        debug=False,
        lr=0.0001,
        metric='val_loss',
        seed=666
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

if __name__ == "__main__":

    makepaths([config.MODEL_PATH, config.PATH])

    train_iterative([0, 1, 2, 3],[4])