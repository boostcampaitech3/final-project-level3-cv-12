import io
from typing import List, Dict, Any

import albumentations
import albumentations.pytorch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image



def get_model_yolov5():
    """Model을 가져옵니다"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
    model.to(device)
    return model


def _transform_image(image_bytes: bytes):
    transform = albumentations.Compose(
        [
            albumentations.Resize(height=1024, width=1024),
            albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            albumentations.pytorch.transforms.ToTensorV2(),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image_array = np.array(image)
    return transform(image=image_array)["image"].unsqueeze(0)


def predict_from_image_byte(model: get_model_yolov5, image_bytes: bytes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # transformed_image = _transform_image(image_bytes).to(device)
    # outputs = model(transformed_image)
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image_array = np.array(image)
    outputs = model(image_array)
    label = outputs.pandas().xyxy
    # return [True for l in label if l == 'Person']
    # return list(outputs.pandas().xyxy[0]['name'])
    return label
