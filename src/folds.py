import pandas as pd
import config
import os

from sklearn.model_selection import KFold

import glob

import numpy as np
import re

from tqdm import tqdm

if not os.path.exists(config.OUTPUT):
    os.makedirs(config.OUTPUT)


df = pd.read_csv(config.TRAIN_CSV)
df.loc[:,'kfold'] = -1

df['x'] = -1
df['y'] = -1
df['w'] = -1
df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

df[['x', 'y', 'w', 'h']] = np.stack(df['bbox'].apply(lambda x: expand_bbox(x)))
df.drop(columns=['bbox'], inplace=True)
df['x'] = df['x'].astype(np.float)
df['y'] = df['y'].astype(np.float)
df['w'] = df['w'].astype(np.float)
df['h'] = df['h'].astype(np.float)


files = glob.glob(f"{config.TRAIN_PATH}/*.jpg")

# print(files)

format = lambda x: os.path.splitext(os.path.basename(x))[0]

files = list(map(format, files))

files = pd.Series(files)

print(files)

names = df.image_id

kf = KFold(n_splits=5, random_state=42, shuffle=True) 
folds = kf.split(files)

for fold, (x, y) in enumerate(folds):
    # print(len(y), fold)
    # print(files.iloc[y])
    for file in tqdm(files.iloc[y].values):
        # print(file)
        z = names[names  == file].index.tolist()
        df.loc[z, 'kfold'] = fold

print(df.kfold.value_counts())
df.to_csv(config.TRAIN_FOLDS, index=False)

print(df.head(10))
print(df.tail(10))

