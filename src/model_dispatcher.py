from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def model_dispenser():

    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    head = FastRCNNPredictor(in_features, num_classes=2)

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = head

    return model
