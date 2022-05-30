# import torch
# import torchvision
# import albumentations as A
# from torch.utils.data import Dataset
# from albumentations.pytorch import ToTensorV2


# class outDetector():

#     def __init__(self):

#         self.test_transform = A.Compose([
#                 A.Resize(384, 384),
#                 A.Normalize(),
#                 ToTensorV2()
#             ])

#         use_cuda = torch.cuda.is_available()
#         self.DEVICE = torch.device("cuda" if use_cuda else "cpu")
#         self.model = torchvision.models.resnet18(pretrained=False)
#         self.model.fc = torch.nn.Linear(512, 3)
#         self.model = self.model.to(self.DEVICE)


#         model_path = '/opt/ml/project/final-project-level3-cv-12/weights/outDetector.pth'
#         ckpt = torch.load(model_path, map_location=self.DEVICE)
#         self.model.load_state_dict(ckpt)


#     def predict(self, img, bboxes):
#         preds = []
#         for bbox in bboxes:
#             (x, y, w, h) = bbox
#             cropped = img[y:y+h, x:x+w]
#             cropped = self.test_transform(image=cropped)['image']
#             if len(cropped.shape) == 3:
#                 cropped = cropped.unsqueeze(0)
#             cropped = cropped.to(self.DEVICE)

#             pred = self.model(cropped)
#             pred = pred.argmax(dim=-1)
#             pred = pred.item()
#             preds.append(pred)
        
#         return preds

# 결품 bbox의 위치가 주어졌을 때,
# before_img에서 양 옆 bbox와의 기울기와
# after_img에서 양 옆 bbox와의 기울기 차이를 구함

### left bbox나 right bbox에 결품이 생겼을 때 기준을 어떻게 정할지 고려해야함

def get_bbox_info(x, y, w, h):
    x_center = x + w//2
    y_center = y + h//2
    x_min, y_min, x_max, y_max = x, y, x+w, y+h
    return x_min, y_min, x_max, y_max, x_center, y_center


def get_side_bbox(bbox, bboxes):
    x_min, y_min, x_max, y_max, x_center, y_center = get_bbox_info(*bbox)

    min_left_distance = 100000
    min_right_distance = 100000

    for bbox2 in bboxes:
        bbox2_info = get_bbox_info(*bbox2)
        x_min2, y_min2, x_max2, y_max2, x_center2, y_center2 = bbox2_info
        cond1 = y_min < y_center2 < y_max  # 범위 체크 1
        cond2 = y_min2 < y_center < y_max2 # 범위 체크 2
        if cond1 and cond2:
            distance = abs(x_center2 - x_center)

            if x_center2 < x_center: #left
                if distance < min_left_distance:
                    min_left_distance = distance
                    min_left = bbox2_info

            elif x_center2 > x_center: #right
                if distance < min_right_distance:
                    min_right_distance = distance
                    min_right = bbox2_info

    return min_left, min_right


def get_slope(bbox, bboxes):
    x_center, y_center = get_bbox_info(*bbox)[-2:]
    min_left, min_right = get_side_bbox(bbox, bboxes)
    left_x, left_y = min_left[-2:]
    right_x, right_y = min_right[-2:]
    left_slope =  -round((left_y - y_center) / (left_x - x_center), 2)
    right_slope =  -round((right_y - y_center) / (right_x - x_center), 2)
    
    return left_slope, right_slope


def check_status(before_bbox, after_bbox, before_bboxes, after_bboxes):
    
    before_left_slope, before_right_slope = get_slope(before_bbox, before_bboxes)
    after_left_slope, after_right_slope = get_slope(after_bbox, after_bboxes)

    d_left = (after_left_slope - before_left_slope) 
    d_right = (after_right_slope - before_right_slope)
    
#     d_left = 0 if -0.05 < d_left < 0.05 else d_left
#     d_right = 0 if -0.05 < d_right < 0.05 else d_right
    
    prod = d_left - d_right
    return prod