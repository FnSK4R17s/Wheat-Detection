import dataset
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Rectangle

import cv2

import config

import pandas as pd
import utils

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)

def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    if config.DATA_FMT == 'pascal_voc':
        x_min, x_max, y_min, y_max = int(x_min), int(w), int(y_min), int(h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    return img


def visualize(annotations, category_id_to_name):
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)


def denorm(image):

    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    mR, mG, mB = config.MODEL_MEAN

    sR, sG, sB = config.MODEL_STD

    R = (R+mR)*sR
    G = (G+mG)*sG
    B = (B+mB)*sB

    img = np.array([])

    img = np.dstack((R, G))
    img = np.dstack((img, B))

    print(img.shape)

    return img
    

# data = dataset.WheatDataset(folds=[0,1,2])
train_df = pd.read_csv(config.TRAIN_FOLDS)
folds=[0,1,2]
train_df = train_df[train_df.kfold.isin(folds)].reset_index(drop=True)

data = dataset.AwgDataset(train_df, config.TRAIN_PATH)

print(len(data))

idx = 122

image, target, img_name = data[idx]
# img, res, img_name = data[idx]

# print(res)

# labels = []
# bboxes = []

# for l, a, b, c, d in res:
#     labels.append(1)
#     bboxes.append([a,b,c,d])



img = image.numpy()
bboxes = target['boxes'].numpy()
labels = target['labels'].numpy()

print(img)
print(img.shape)
print(bboxes.shape)

img = np.transpose(img, (1, 2, 0))

img = denorm(img)

annotations = {'image': img, 'bboxes': bboxes, 'category_id': labels}

category_id_to_name = {1: 'wheat'}

visualize(annotations, category_id_to_name)

print(img_name)

plt.show()

del plt