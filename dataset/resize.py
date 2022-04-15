import os
import json

import cv2
import numpy as np
import pandas as pd 
from glob import glob
from tqdm import tqdm
from PIL import Image
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rc('font', family='NanumGothicCoding')


class Info:
    def __init__(self):
        self.train = self.get_data('train.json')
        self.val = self.get_data('val.json')
    
    def get_data(self, f_name):
        with open(f_name, mode='r') as f:
            _json = json.load(f)
        ann = pd.DataFrame(_json['annotations'])
        img = pd.DataFrame(_json['images'])
        df = pd.merge(ann, img, left_on='image_id', right_on='id').drop(['id_y'], axis=1).rename(columns={'id_x':'id'})
        return df
    
    def plot(self, img_id):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.axis('off')

        try:
            self._plot(self.train, img_id, ax)
            fig.suptitle(f'"{img_id}" is in train', size=20)
        except:
            self._plot(self.val, img_id, ax)
            fig.suptitle(f'"{img_id}" is in val', size=20)

    def _plot(self, _df, img_id, ax):
        df = _df.query(f'image_id == "{img_id}"')
        assert len(df), 'this "image id" is not in dataframe'
        path = df['file_name'].values[0]
        img = np.array(Image.open(path))
        ax.imshow(img)
        for bbox in df['bbox']:
            ax.add_patch(patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3], edgecolor='seagreen', 
                fill=False, lw=3
            ))
        ax.set_title(df['file_name'].values[0].split('/')[-2], size=20)


def resize(df, re_width=1024, re_height=1024):

    img2boxes = df.groupby('file_name')['bbox'].apply(list)
    
    re_df = df.copy().drop(['bbox', 'width', 'height', 'area'], axis=1)
    
    re_df['width'] = re_width
    re_df['height'] = re_height

    transform = A.Compose([
        A.Resize(height=re_height, width=re_width)],
        bbox_params=A.BboxParams(
            format='coco', 
            min_visibility=0.1, 
            label_fields=[]
            )
        )
    for img_path, bboxes in tqdm(img2boxes.iteritems(), total=len(img2boxes)):
        img = cv2.imread(img_path)

        # transform
        transformed = transform(image=img, bboxes=bboxes)
        re_img, re_bboxes = transformed['image'], transformed['bboxes']
        re_df.loc[re_df['file_name']  == img_path, 'bbox'] = pd.Series(re_bboxes).values
        
        os.makedirs('re'+'/'.join(img_path.split('/')[:-1]), exist_ok=True)
        cv2.imwrite('re'+img_path, re_img)

    re_df['area'] = re_df['bbox'].apply(lambda x: round(x[2]*x[3], 2))

    return re_df


def make_json(json_path, df):
    with open(json_path, mode='r') as f:
        _json = json.load(f)

    _json['images'] = []
    _json['annotations'] = []

    for _, row in df.iterrows():
        _json['images'].append({
            'file_name': 're'+row['file_name'],
            'height': row['height'],
            'width': row['width'],
            'id': row['image_id']
        })
        _json['annotations'].append({
            'id' : row['id'],
            'image_id' : row['image_id'],
            'bbox':row['bbox'],
            'area' : row['area'],
            'category_id':row['category_id'],
            'iscrowd' : 0,
        })
    with open('re'+json_path, mode='w') as f:
        json.dump(_json, f)


if __name__ == "__main__":
    info = Info()
    re_train = resize(info.train, re_width=1024, re_height=1024)
    make_json('train.json', re_train)

    re_val = resize(info.val, re_width=1024, re_height=1024)
    make_json('val.json', re_val)