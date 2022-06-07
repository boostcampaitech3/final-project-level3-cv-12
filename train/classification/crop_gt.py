import os
import pandas as pd
from PIL import Image
from PIL import Image, ImageOps
from argparse import ArgumentParser
from pycocotools.coco import COCO

def crop(folder_path, save_path, json_path):
    
    coco = COCO(json_path)
    annotation_id = coco.getAnnIds()

    num = 0 #이미지 이름
    image_name = []
    label_name = []
    for idx in range(len(annotation_id)):
        annotation_info = coco.loadAnns(idx+1)[0]
        cate_id = annotation_info['category_id']

        file_path = coco.loadImgs(annotation_info['image_id'])[0]['file_name']
        image_path = os.path.join(folder_path, file_path)
    
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        x, y, w, h = annotation_info['bbox']
        crop_image = image.crop((x,y,x+w,y+h))
        if crop_image.width == 0 or crop_image.height == 0:
            continue
        crop_image.save(f"{save_path}/{num}.jpg")
        image_name.append(f"{num}.jpg")
        label_name.append(cate_id-1)
        num += 1

    gt_data = {'image_id':image_name, 'classes':label_name}
    df = pd.DataFrame(gt_data)
    df.to_csv(f"{save_path}/crop.csv",index=False)


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--folder_path', type=str, default = '/opt/ml/project/final-project-level3-cv-12/dataset/images', help='image folder path')
    parser.add_argument('--save_path', type=str, default = '/opt/ml/project/final-project-level3-cv-12/dataset/gt/val', help='save folder path')
    parser.add_argument('--json_path', type=str, default = '/opt/ml/project/final-project-level3-cv-12/dataset/instances_Validation.json', help='json file path')

    args = parser.parse_args()

    return args

def main(args):
    crop(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)