import numpy as np

def collate_fn(batch):
    return tuple(zip(*batch))

def bboxtoIoU(results, targets):
    IoU = []
    for i, (res, gt) in enumerate(zip(results, targets)):
        b_res = res['boxes'].cpu().detach().numpy()
        b_gt = gt['boxes'].cpu().detach().numpy()

        score = run(b_res, b_gt)
        mean = np.mean(np.max(score, axis=0))
        IoU.append(mean)
    return IoU
        
def run(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea + 1e-2)
    return iou

def format_prediction_string(boxes, scores):
    pred_strings = []
    for s, b in zip(scores, boxes.astype(int)):
        pred_strings.append(f'{s:.4f} {b[0]} {b[1]} {b[2] - b[0]} {b[3] - b[1]}')

    return " ".join(pred_strings)