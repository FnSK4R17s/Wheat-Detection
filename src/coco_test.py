# %matplotlib inline
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
# import skimage.io as io
# import pylab
# pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import config

#initialize COCO ground truth api
def coco_results():
    
    annFile = config.annFile
    cocoGt=COCO(annFile)

    #initialize COCO detections api

    resFile=config.resFile
    cocoDt=cocoGt.loadRes(resFile)

    imgIds=sorted(cocoGt.getImgIds())
    imgIds=imgIds[0:100]
    imgId = imgIds[np.random.randint(100)]

    # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    cocoEval.params.imgIds  = imgIds
    cocoEval.params.iouThrs = np.linspace(.5, 0.75, np.round((0.75 - .5) / .05) + 1, endpoint=True)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()