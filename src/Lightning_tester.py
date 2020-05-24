import pytorch_lightning as pl

from model_dispatcher import MODEL_DISPATCHER
from Lightning_module import LitWheat, WheatTest

import config
import torch

import pandas as pd

import numpy as np
import cv2
import matplotlib.pyplot as plt

def collate_fn(batch):
    return tuple(zip(*batch))


def test_dataloader():
    test_loader = torch.utils.data.DataLoader(WheatTest(),
                                               batch_size=config.TEST_BATCH_SIZE,
                                               shuffle=False,
                                               collate_fn=collate_fn)

    return test_loader

def format_prediction_string(boxes, scores):
    pred_strings = []
    for s, b in zip(scores, boxes.astype(int)):
        pred_strings.append(f'{s:.4f} {b[0]} {b[1]} {b[2] - b[0]} {b[3] - b[1]}')

    return " ".join(pred_strings)

def inference():

    train_folds = [0, 1, 2, 3]
    valid_folds = [4]

    model = MODEL_DISPATCHER[config.MODEL_NAME]
    model.load_state_dict(torch.load(config.MODEL_SAVE))

    # lit_model = LitWheat.load_from_checkpoint(checkpoint_path=config.PATH, model=model, train_folds=train_folds,  valid_folds=valid_folds)

    # torch.save(lit_model.model.state_dict(), config.MODEL_SAVE)

    # trainer = pl.Trainer(gpus=1)
    # trainer.test(lit_model)

    # lit_model.test_df.to_csv(config.SUB_FILE, index=False)
    # print(lit_model.test_df.head())

    # model.load_state_dict(torch.load(config.MODEL_SAVE))
     

    model = model.to(config.DEVICE)

    model.eval()

    test_df = pd.DataFrame(columns=['image_id', 'PredictionString'])

    with torch.no_grad():

        test_data_loader = test_dataloader()

        for images, _, img_name in test_data_loader:
            images = list(image.to(config.DEVICE) for image in images)

            results = []
            outputs = model(images)
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
            test_df = test_df.append(df, ignore_index=True)

    im_id = 1

    sample = images[im_id].permute(1,2,0).cpu().numpy()
    boxes = outputs[im_id]['boxes'].data.cpu().numpy()
    scores = outputs[im_id]['scores'].data.cpu().numpy()

    boxes = boxes[scores >= config.detection_threshold].astype(np.int32)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in boxes:
        cv2.rectangle(sample,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (220, 0, 0), 2)
        
    ax.set_axis_off()
    ax.imshow(sample)

    plt.show()


if __name__ == "__main__":
    inference()