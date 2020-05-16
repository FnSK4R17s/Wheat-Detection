import pytorch_lightning as pl

from model_dispatcher import model_dispenser
from Lightning_module import LitWheat

import config
import torch

def inference():

    train_folds = [0, 1, 2, 3]
    valid_folds = [4]

    model = model_dispenser()
    model.load_state_dict(torch.load(config.MODEL_PATH))

    lit_model = LitWheat(train_folds,  valid_folds, model=model)

    trainer = pl.Trainer()
    trainer.test(lit_model)


if __name__ == "__main__":
    inference()