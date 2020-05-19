from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


import torch.nn as nn

import config

import torch
import numpy as np
from PIL import Image


def model_dispenser():

    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    head = FastRCNNPredictor(in_features, num_classes=config.CLASSES)

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = head

    return model

def FasterRCNN_mobileNet():

    # net = torchvision.models.mobilenet_v2(pretrained=True).features
    # modules = list(net.children())[:-2]
    # backbone = nn.Sequential(*modules)

    backbone = torchvision.models.mobilenet_v2(pretrained=True).features

    # test_backbone(backbone)

    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone,
                    num_classes=config.CLASSES,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)



def FasterRCNN_resnext50_32x4d():
    
    net = torchvision.models.resnext50_32x4d(pretrained=True)
    modules = list(net.children())[:-1]
    backbone = nn.Sequential(*modules)

    # backbone = torchvision.models.resnext50_32x4d(pretrained=True).features

    # test_backbone(backbone)

    backbone.out_channels = 2048

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone,
                    num_classes=config.CLASSES,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)


    return model

def FasterRCNN_resnext101_32x8d():
    
    net = torchvision.models.resnext101_32x8d(pretrained=True)
    modules = list(net.children())[:-2]
    backbone = nn.Sequential(*modules)

    # test_backbone(backbone)

    backbone.out_channels = 2048

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone,
                    num_classes=config.CLASSES,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)


    return model

def FasterRCNN_resnet101():
    backbone = resnet_fpn_backbone('resnet101', pretrained=True)
    # net = torchvision.models.resnet101(pretrained=True)
    # modules = list(net.children())[:-2]
    # backbone = nn.Sequential(*modules)

    # backbone = torchvision.models.resnet101(pretrained=True).features

    test_backbone(backbone)

    backbone.out_channels = 256

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone,
                    num_classes=config.CLASSES,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)


    return model


def vgg():

    vgg = torchvision.models.vgg16(pretrained=True)
    backbone = vgg.features[:-1]
    for layer in backbone[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    backbone.out_channels = 512
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)

    class BoxHead(nn.Module):
        def __init__(self, vgg):
            super(BoxHead, self).__init__()
            self.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        def forward(self, x):
            x = x.flatten(start_dim=1)
            x = self.classifier(x)
            return x
    box_head = BoxHead(vgg)


    model = torchvision.models.detection.faster_rcnn.FasterRCNN(
            backbone, #num_classes,
            rpn_anchor_generator = anchor_generator,
            box_roi_pool = roi_pooler,
            box_head = box_head,
            box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(4096, num_classes=1))
    
    return model

def test_backbone(backbone):
    
    x = f'{config.TRAIN_PATH}/0a3cb453f.jpg'


    image = np.array(Image.open(x))

    image = np.transpose(image, (2, 0, 1)).astype(np.float32)

    image = torch.tensor(image, dtype=torch.float)

    print(image.size())

    image.unsqueeze_(0)

    print(image.size())


    backbone.eval()
    with torch.no_grad():
        y = backbone(image)

    print(y)
    print(y['0'].shape)

    backbone.train()

MODEL_DISPATCHER = {
    'FasterRCNN_mobileNet' : FasterRCNN_mobileNet(),
    'FasterRCNN_resnext' : FasterRCNN_resnext50_32x4d(),
    'FasterRCNN_resnext101' : FasterRCNN_resnext101_32x8d(),
    'FasterRCNN_resnet101' : FasterRCNN_resnet101(),
    'fasterRCNNresnet50' : model_dispenser(),
    'fasterRCNNVGG' : vgg()
}

