# ResNet18

<br>

## 1. Train & Validation

<br>

### 1) crop_gt.py
- 원본 이미지에서 object bbox 정보를 받아 crop 후 각 image와 csv 저장
```
python crop_gt.py --folder_path {원본 이미지가 있는 폴더 경로} --save_path {crop 이미지와 csv 저장 폴더 경로} --json_path {이미지 정보 담긴 json 파일 경로}
```

<br>

### 2) train_gt.py
- crop 이미지 이용 train & validation
- validation loss가 작을 때 weight 파일 저장
- validation 결과 csv 파일로 저장

```
python train_gt.py --seed {seed number} --train_path {train_csv 파일 경로} --val_path {validation_csv 파일 경로} --folder_path {crop 이미지 폴더가 있는 폴더 경로} --saved_dir {weight 파일 저장 폴더 경로} --img_size {image size (height, width)} --epoch {num epoch}
```
<br>

### 2-1) crop 없이 train
- crop 없이 원본 이미지 train -> data loading 속도가 느려서 이용 x
```
python train_no_crop.py --seed {seed number} --train_path {train_json 파일 경로} --val_path {validation_json 파일 경로} --folder_path {이미지 폴더가 있는 폴더 경로} --saved_dir {weight 파일 저장 폴더 경로} --img_size {image size (height, width)} --epoch {num epoch}
```

<br>

## 2. Inference

<br>

### 1) result_crop_image.py
- yolov5가 detect한 bbox 정보를 이용해 crop 후 각 image와 csv 저장
```
python result_crop_image.py --folder_path {crop image 저장 폴더 경로} --image_path {test 이미지 파일 경로} --text_path {yolov5 detect 결과 생성되는 txt 파일 경로} --confidence {bbox confidence score 최소값}
```

<br>

### 2) inference.py
- test dataset inference
- inference 결과 csv 파일로 저장

```
python inference.py --model_path {weight 파일 경로} --test_dir {test data 폴더 경로}
```
<br>
