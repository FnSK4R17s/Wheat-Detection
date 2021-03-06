CROP_SIZE = 512
CHANNELS = 3
CLASSES = 2

DEVICE = 'cuda'

EPOCHS = 7

LR = 0.0003

INPUT_SHAPE = (CHANNELS, CROP_SIZE, CROP_SIZE)

MODEL_NAME = 'fasterRCNNresnet50'


min_area = 0.0
min_visibility = 0.0

INPUT = 'input'
OUTPUT = 'output'

TRAIN_PATH = f'{INPUT}/train'
TEST_PATH = f'{INPUT}/test'

TRAIN_CSV = f'{INPUT}/train.csv'
TEST_CSV = f'{INPUT}/sample_submission.csv'

MODEL_MEAN = (0.485, 0.456, 0.406)
MODEL_STD = (0.229, 0.224, 0.225)

TRAIN_FOLDS = f'{OUTPUT}/train_folds.csv'

BATCH_SIZE = 4


logs_path = 'lightning_logs'
version = 'version_13'
ckpt_name = 'epoch=4'

PATH = f'{logs_path}/{version}/checkpoints/{ckpt_name}.ckpt'

DATA_FMT = 'pascal_voc'

MODEL_PATH = f'saved_models/{MODEL_NAME}'
# MODEL_PATH = f'saved_models'
MODEL_SAVE = f'{MODEL_PATH}/model.pth'

FILEPATH = 'saved/'

FILESAVE = f'{FILEPATH}/model.pth'

detection_threshold = 0.5

SUB_FILE = f'{OUTPUT}/submission.csv'

annFile = 'COCO/ann_real.json'
resFile = 'COCO/ann_fake.json'

ACCUMULATE = 32