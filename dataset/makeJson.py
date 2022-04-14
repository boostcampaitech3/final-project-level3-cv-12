import json
from glob import glob
from tqdm import tqdm
import xml.etree.ElementTree as elemTree

def make_json():

    jsons = {}

    for data_type in ['Training', 'Validation']:
        temp_json = {
            'images':[],
            'categories':[],
            'annotations':[]
        }

        label_folders = sorted(glob(f'{data_type}/label/*'))
        
        #categories
        for _id, cat in enumerate(label_folders):
            cat = cat.split('/')[-1]
            temp_json['categories'].append(
                {
                    'supercategory':'Defect',
                    'id': _id,
                    'name':cat
                }
            )

        # images, annotations
        ann_id = 0
        for _id, label_folder in tqdm(enumerate(label_folders), total=len(label_folders)):
            label_files = sorted(glob(label_folder + '/*'))
            for label_file in label_files:
                if 'meta' in label_file:
                    f_name = label_file\
                        .replace('label', 'image')\
                        .replace('_meta.xml', '.jpg')
                    xml = open(label_file, 'rt', encoding='UTF8')
                    tree = elemTree.parse(xml)
                    root = tree.getroot()
                    item_no = root.find('div_cd').find('item_no').text
                    annotation = root.find('annotation')
                    size = annotation.find('size')
                    w = size.find('width').text
                    h = size.find('height').text
                    img_id = f_name.split('/')[-1].split('.')[0]
                    
                    temp_json['images'].append({
                        'file_name':f_name,
                        'height': int(h),
                        'width': int(w),
                        'id': img_id
                    })
                    for _object in annotation.findall('object'):
                        ann_id +=1
                        bbox = _object.find('bndbox')
                        xmin = float(bbox.find('xmin').text)
                        ymin = float(bbox.find('ymin').text)
                        xmax = float(bbox.find('xmax').text)
                        ymax = float(bbox.find('ymax').text)
                        ann_w = xmax-xmin
                        ann_h = ymax-ymin
                        temp_json['annotations'].append({
                            'id' : ann_id,
                            'image_id' : img_id,
                            'bbox':[xmin, ymin, ann_w, ann_h],
                            'area' : ann_w * ann_h,
                            'category_id':_id,
                            'is_crowd' : 0,
                            'segmentation' : 0
                        })
        jsons[data_type] = temp_json
    
    return jsons


def save_json(_jsons):
    with open('train.json', mode='w') as f:
        json.dump(_jsons['Training'], f)
    with open('val.json', mode='w') as f:
        json.dump(_jsons['Validation'], f)

if __name__ == "__main__":
    jsons = make_json()
    save_json(jsons)
