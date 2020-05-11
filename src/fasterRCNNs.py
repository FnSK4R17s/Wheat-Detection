import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import torch.nn as nn
from torch.nn import functional as F

import config

import pretrainedmodels

class FasterRCNNResNet18(nn.Module):
    def __init__(self, pretrained):
        super(FasterRCNNResNet18, self).__init__()
        if pretrained is True:
            base_model = pretrainedmodels.__dict__['resnet18'](pretrained='imagenet')
        else:
            base_model = pretrainedmodels.__dict__['resnet18'](pretrained=None)

        backbone = base_model.features
        backbone.out_channels = 512

        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)

        self.model = FasterRCNN(base_model,
                   num_classes=config.num_classes,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

    def forward(self, x):
        return self.model(x)



class FasterRCNNResNet34(nn.Module):
    def __init__(self, pretrained):
        super(FasterRCNNResNet34, self).__init__()
        if pretrained is True:
            base_model = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')
        else:
            base_model = pretrainedmodels.__dict__['resnet34'](pretrained=None)

        backbone = base_model.features
        backbone.out_channels = 512

        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)

        self.model = FasterRCNN(base_model,
                   num_classes=config.num_classes,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

    def forward(self, x):
        return self.model(x)