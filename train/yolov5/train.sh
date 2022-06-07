#!/bin/bash

python train.py --weights /opt/ml/project/final-project-level3-cv-12/yolov5/ramen/test/weights/best.pt --data /opt/ml/project/final-project-level3-cv-12/yolov5/data/ramen.yaml --hyp /opt/ml/project/final-project-level3-cv-12/yolov5/data/hyps/hyp.p6.yaml --epochs 1500 --imgsz 1024 --batch-size 12 --project ramen --name all_test --optimizer Adam --entity yolo12