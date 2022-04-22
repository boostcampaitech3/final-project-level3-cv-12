import json
from glob import glob
from tqdm import tqdm
import xml.etree.ElementTree as elemTree
from PIL import Image

def make_json():

    temp_json = {
            'images':[],
            'categories':[],
            'annotations':[]
    }

    classes = ['neogurimild',
    'neogurispicy',
    'neoguriangry',
    'gamjamyeon',
    'jjawang',
    'jjawangspicy',
    'doongjibibim',
    'doongjidongchimi',
    'jinramenmild',
    'jinramenspicy']

    # categories
    for _id, cat in enumerate(classes):
        temp_json['categories'].append(
            {
                'supercategory':'Defect',
                'id': _id,
                'name':cat
            }
        )
    
    ann_id = 0

    label_folders = sorted(glob('/opt/ml/new/final-project-level3-cv-12/dataset/Intuworks/dataset6_myeon/label/*.txt')) # fix directory
    for label_path in label_folders:
        # images
        id_name = label_path.split('/')[-1].split('.')[0]
        img_path = label_path.split('.')[0] + '.jpg'
        img = Image.open(img_path)
        width, height = img.size
        temp_json['images'].append({
                    'file_name':img_path,
                    'height': int(height),
                    'width': int(width),
                    'id': id_name
                    })
        
        #annotations
        labels = open(label_path).read().splitlines()
        for label in labels:
            c, x, y, w, h = map(float, label.split())
            c = int(c)
            x = width * x
            y = height * y
            w = width * w
            h = height * h
            xmin, ymin = x-w//2, y-h//2
            temp_json['annotations'].append({
                    'id' : ann_id,
                    'image_id' : id_name,
                    'bbox':[xmin, ymin, w, h],
                    'area' : w * h,
                    'category_id': c,
                    'iscrowd' : 0,
                    })
            ann_id += 1

    with open('train.json', mode='w') as f:
        json.dump(temp_json, f)

if __name__ == "__main__":
    jsons = make_json()
