import pytorch_lightning as pl

from model_dispatcher import model_dispenser
from Lightning_module import LitWheat

import config
import torch

def inference():

    train_folds = [0, 1, 2, 3]
    valid_folds = [4]

    model = model_dispenser()
    # model.load_state_dict(torch.load(config.MODEL_PATH))

    lit_model = LitWheat.load_from_checkpoint(checkpoint_path=config.PATH, model=model, train_folds=train_folds,  valid_folds=valid_folds)

    trainer = pl.Trainer()
    trainer.test(lit_model)

    lit_model.test_df.to_csv(config.SUB_FILE, index=False)
    print(lit_model.test_df.head())


if __name__ == "__main__":
    inference()