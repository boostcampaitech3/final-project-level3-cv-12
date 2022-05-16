## mmdetection 환경세팅

<br>

### 1) mmdetection 관련 package 설치
```bash
sh setup.sh
```
- 마지막에 `gpu_check`가 `True`가 나오는 것을 확인해주세요

<br>

### 2) 기타 package 설치 (albumentations, pandas, wandb,...)
```bash
pip install -r requirements.txt
```

## yolov5 사용법

<br>

### 1) train.py
```
python train.py --weights {pretraiend된 wieght 파일} --data ramen.yaml --hyp {argument file(hyp.p6.yaml)} --epochs 80 --batch_size 12 --img_size 1024 --project {wandb project 명}
```
- [pretrained 파일](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5x6.pt)
- yolov5/data/ramen.yaml
- yolov5/data/hyps/hyp.p6.yaml

<br>

### 2) detect.py

```
python detect.py --weights {best weight 파일} --source {test dataset 위치} --imgsz 1024 --data ramen.yaml --conf-thres 0.08 --iou-thres 0.5 --name {폴더명} --save-txt --save-conf
```
<br>

### 3) 데이터 구조

- yolo format normalized[x_center,y_center,width,height]  

train/data,label
valid/data,label
<br>
ramen.yaml 참고

```bash
data
├── data

├── train
│   ├── data
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   ├── 0003.jpg
│   │   └── ...
│   ├── label
│   │   ├── 0001.txt
│   │   ├── 0002.txt
│   │   ├── 0003.txt
│   │   └── ...
├── valid
│   ├── data
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   ├── 0003.jpg
│   │   └── ...
│   ├── label
│   │   ├── 0001.txt
│   │   ├── 0002.txt
│   │   ├── 0003.txt
│   │   └── ...

``` 

