import torch
import cv2
import numpy as np

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import check_img_size , non_max_suppression,scale_coords,xyxy2xywh
from yolov5.utils.datasets import LoadImages
from pathlib import Path

from yolov5.utils.augmentations import letterbox


def ramen_detect(im0s):
    weights = "/opt/ml/project/final-project-level3-cv-12/weights/ramen_best.pt"
    device = torch.device("cpu")
    data = "/opt/ml/project/final-project-level3-cv-12/yolov5/data/ramen.yaml"
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt

    height, width = im0s.shape[:2]

    imgsz = check_img_size((1440,1440), s=stride)

    img = letterbox(im0s, imgsz, stride=stride, auto=True)[0]
    img = img.transpose((2,0,1))[::-1]
    img = np.ascontiguousarray(img)

    im = torch.from_numpy(img).to(device)
    im = im.float()
    im /= 255
    im = im[None]

    
    pred = model(im)
    pred = non_max_suppression(pred,0.7,0.7,None,False,False,max_det=1000)

    im0 =  im0s.copy()
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
    
    
    output = []

    for i,det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                if True:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if True else (cls, *xywh)  # label format
                    x = []
                    for i in line:
                        x.append(round(float(i),5))
                    x[0] = int(x[0])
                    _cls, _xc, _yc, _w, _h, conf = x
                    _x = (_xc - _w/2) * width
                    _y = (_yc - _h/2) * height
                    _w *= width
                    _h *= height

                    output.append([int(_x), int(_y), int(_w), int(_h)])

    return output


