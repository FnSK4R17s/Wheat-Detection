import pytorch_lightning as pl
from Lightning_module import LitWheat
from model_dispatcher import model_dispenser
from pytorch_lightning.loggers import TensorBoardLogger


def train_iterative(train_folds,  valid_folds):
    model = model_dispenser()
    lit_model = LitWheat(train_folds,  valid_folds, model=model)

    early_stopping = pl.callbacks.EarlyStopping(mode=max, monitor='metric', patience=50)
    model_checkpoint = pl.callbacks.ModelCheckpoint(filepath='saved_models/', monitor='metric', save_top_k=1)

    tb_logger = TensorBoardLogger(save_dir='logs/')

    trainer = pl.Trainer(
        gpus=1,
        accumulate_grad_batches=64,
        profiler=True,
        logger=[tb_logger],
        early_stop_callback=early_stopping,
        checkpoint_callback=model_checkpoint,
        gradient_clip_val=0.5,
        debug=False,
        lr=0.0001,
        metric='metric',
        seed=666
    )

    trainer.fit(lit_model)

if __name__ == "__main__":
    train_iterative([0, 1, 2, 3],[4])