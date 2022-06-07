import os
import pandas as pd
from PIL import Image
from PIL import Image, ImageOps
from argparse import ArgumentParser

def crop(folder_path, image_path, text_path, confidence):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    width, height = image.size

    folder_name = image_path.split('/')[-1].split('.')[0]
    folder_list = os.listdir(folder_path)
    n = 1
    while True:
        if folder_name in folder_list:
            n += 1
            folder_name += str(n)
        else:
            save_path = os.path.join(folder_path, folder_name)
            os.mkdir(save_path)
            break

    num = 0 #이미지 이름

    image_name = []

    with open(text_path) as f:
        lines = f.readlines()
        for line in lines:
            cls, xc, yc, w, h, conf = map(float, line.split())
            if conf >= confidence:
                x = xc - w/2
                y = yc - h/2
                x *= width
                w *= width
                y *= height
                h *= height
                crop_image = image.crop((x,y,x+w,y+h))
                crop_image.save(f"{save_path}/{num}.jpg")
                image_name.append(f"{num}.jpg")
                num += 1

    df = pd.DataFrame(image_name, columns=['image_id'])
    df.to_csv(f"{save_path}/crop.csv",index=False)


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--folder_path', type=str, default = '/opt/ml/test/yolo_crop', help='crop image save path')
    parser.add_argument('--image_path', type=str, default = '/opt/ml/final-project-level3-cv-12/dataset/Test/KakaoTalk_20220519_165405194_01.jpg', help='image file path')
    parser.add_argument('--text_path', type=str, default = '/opt/ml/final-project-level3-cv-12/train/yolov5/runs/detect/exp/labels/KakaoTalk_20220519_165405194_01.txt', help='txt file path')
    parser.add_argument('--confidence', type=float, default=0.75, help='confidence score')

    args = parser.parse_args()

    return args

def main(args):
    crop(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)