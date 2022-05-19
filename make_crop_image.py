import json
import pandas as pd
from tqdm import tqdm
from PIL import Image

with open("/opt/ml/test/annotations/instances_Train.json") as f:
    json_data = json.load(f)

x_num = [18,63,2,27,43,69,16,61,67,1,106,12,20,5,70,26,62,88,29,36] #상위 20개 라면 index


image_name = []
index_num = []


path = '/opt/ml/test/images/' #이미지 경로
save_path = '/opt/ml/test/crop_image/'  #저장 경로
num = 0 #이미지 이름
for i in tqdm(range(len(json_data['annotations']))):
    if i in [8704,9410]:
        continue
    x,y,w,h = map(int,json_data['annotations'][i]['bbox'])
    if json_data['images'][json_data['annotations'][i]['image_id']-1]['file_name'].startswith("2"):
        image = Image.open(path + json_data['images'][json_data['annotations'][i]['image_id']-1]['file_name']).convert("RGB").rotate(270)    
    else:
        image = Image.open(path + json_data['images'][json_data['annotations'][i]['image_id']-1]['file_name']).convert("RGB")
    crop_image = image.crop((x,y,x+w,y+h))
    crop_image.save(f"{save_path}{num}.jpg")

    image_name.append(f"{num}.jpg")
    
    if json_data['annotations'][i]['category_id'] in x_num:
        index_num.append(x_num.index(json_data['annotations'][i]['category_id']))
    else:
        index_num.append(20)   #other 처리

    num += 1

df = pd.DataFrame((zip(image_name,index_num)),columns=['image_id','classes'])
df.to_csv("/opt/ml/test/crop.csv",index=False)