import fasterRCNNs

import config

MODELS = {
    'fasterRCNNresnet18' : fasterRCNNs.FasterRCNNResNet18(pretrained=True),
    'fasterRCNNresnet34' : fasterRCNNs.FasterRCNNResNet34(pretrained=True)
}