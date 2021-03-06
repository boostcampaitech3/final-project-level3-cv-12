# π λ§€λ κ²°ν κ°μ§ μμ€ν

## π€Ό ν μκ°

CV-12 TEAM YOLO


|κΉμΉν_T3038|λΈμ°½ν_T3074|μ΅μ©μ_T3219|μ΅μ§μ_T3225|μ΅νλ‘_T3228|
|:-:|:-:|:-:|:-:|:-:|
|<a href="https://github.com/KimSeungHyun1217"><img src="imgs/KSH.jpg" width='300px'></a>|<a href="https://github.com/Rohchanghyun"><img src="imgs/NCH.jpg" width='300px'></a>|<a href="https://github.com/chldyddnjs"><img src="imgs/CYW.png" width='300px'></a>|<a href="https://github.com/mango-jina"><img src="imgs/CJA.jpg" width='300px'></a>|<a href="https://github.com/ChoiHongrok"><img src="imgs/CHR.jpg" width='300px'></a>|


## πΆοΈ νλ‘μ νΈ μκ°

<img src="imgs/Untitled.png" width='300px'>

- μ¬κ³ νμ μλν λ° νλ‘μΈμ€ ν¨μ¨νλ₯Ό μν Computer Vision μμ€ν
- λͺ¨λ  λ§€μ₯μμ λ§€λ νλͺ©μ κ΄λ¦¬νλ μμ€νμΌλ‘ μ¬μ©ν  μ μμ

### πΊ Demo

<p align="center">
    <img src="imgs/demo.gif" width = "600px">
</p>

[MORE DEMO](https://www.youtube.com/channel/UCkP9pf52Y9iGt3Goi099a5g/videos)

### π Test

```python
Python Video_test.py --video_name {TestVideo κ²½λ‘} --output_name {Output Video name} --topdown {bool}
```

### π Dataset

**Train** / **Validation Image**

- [Train](https://drive.google.com/drive/folders/1ovW3LX2MdJcSPUlFFlB9IXNZWKJFzAdD?usp=sharing)
- [Validation](https://drive.google.com/drive/folders/1rymW4U1QRRV242O1o2wnktA7cUUt_Eqo?usp=sharing)

**Test**

- [Image](https://drive.google.com/drive/folders/1qmnL2lf2FHrFvSTaEln-eABApLqQfiPQ?usp=sharing)
- [Test_Video](https://drive.google.com/drive/folders/1bq-sye22NnQTo-bNx3IOmUKYyZrtXRLm?usp=sharing)

### Weight
- [YOLOv5](https://drive.google.com/file/d/1pmJZ_5UbWJcqj6xGYBRorGtYFkHIhrt8/view?usp=sharing) 
- [ResNet18](https://drive.google.com/file/d/1h1UMPfn0gm522343xg1XO2A6V4Veotjz/view?usp=sharing) 

### π€ Model

<img src="imgs/Untitled1.png" width='300px'>

| Task | Model | Performance |
| --- | --- | --- |
| Object detection | YOLOv5 | mAP50: 0.9915 |
| Classification | ResNet18 | accuracy: 0.966 |
| κ²°νκ°μ§ | YOLO+RESNET18+SSIM | mAP50: 0.45 |

### ποΈ System Architecture

![KakaoTalk_20220603_201716217.jpg](imgs/KakaoTalk_20220603_201716217.jpg)

## ποΈ Work directory

```bash
|-- README.md
|-- Video_test.py
|-- backend
|   |-- Send_To_Video.py
|   |-- backend_flask_detector.py
|-- dataset
|   |-- README.md
|   |-- images
|   |-- instances_Test.json
|   |-- instances_Train.json
|   |-- instances_Validation.json
|-- evaluation
|   |-- mAP.ipynb
|-- front
|   |-- front_table.py
|   |-- result.csv
|-- modules
|   |-- Human_detect.py
|   |-- calcIou.py
|   |-- changeDetector.py
|   |-- classification.py
|   |-- model.py
|   |-- outDetection.py
|   |-- ramen_detect.py
|   |-- rgb_simularity.py
|   |-- utils.py
|-- requirements.txt
|-- train
|   |-- classification
|   |-- yolov5
|-- weights
|   |-- classification_all.pth
|   |-- outDetector.pth
|   |-- ramen_best.pt
```

## βοΈ Requirements

```bash
yolov5==6.1.2
streamlit==1.9.0
streamlit-webrtc==0.37.0
torchvision==0.8.1
scikit-image==0.19.2
albumentations==1.1.0
flask==2.1.2
torch==1.11.0
uvicorn==0.17.6
```
