import pandas as pd
import config
import os

from sklearn.model_selection import KFold

import glob


df = pd.read_csv(config.TRAIN_CSV)
df.loc[:,'kfold'] = -1

files = glob.glob(f'{config.TRAIN_PATH}\*.jpg')

# print(files)

format = lambda x: os.path.splitext(os.path.basename(x))[0]

files = list(map(format, files))

files = pd.Series(files)

# print(files)

names = df.image_id

kf = KFold(n_splits=5, random_state=42, shuffle=True) 
folds = kf.split(files)

for fold, (x, y) in enumerate(folds):
    # print(len(y), fold)
    # print(files.iloc[y])
    for file in files.iloc[y].values:
        # print(file)
        z = names[names  == file].index.tolist()
        df.loc[z, 'kfold'] = fold

print(df.kfold.value_counts())
df.to_csv(config.TRAIN_FOLDS, index=False)

print(df.head(10))
print(df.tail(10))

