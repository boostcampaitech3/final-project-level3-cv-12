from flask import Flask, request
import numpy as np
import cv2
import torch
import random
import pandas as pd


import modules.model as model
import modules.changeDetector as changeDetector

app = Flask(__name__)

first = True
human = False
bbox = None
# 모델 불러오기
model_flow = model.stockChecker()
check_image = None
ramen = {}

after_image = None

@app.route('/',methods=['POST'], strict_slashes=False)
def hello():
   global first
   global bbox
   global check_image
   global ramen
   global after_image

   human = False
   nparr = np.frombuffer(request.files['image'].read(), np.uint8)
   image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   print(image.shape)
   
   if first:
      output = model_flow.ramen_detect(image)
      for i in output:
         check_label = model_flow.ramen_class(image,[i])
         if check_label[0] in ramen.keys():
               ramen[check_label[0]]["base"] += 1
         else:
               ramen[check_label[0]] = {"base" : 1 , "diff" : 0}
      check_image = image.copy()
      first = False
      ramen = pd.DataFrame(ramen).transpose()

   # if not human:
   #    human = model_flow.check_human(image)

   if not model_flow.check_human(image):
      if after_image:
         bbox = None
         after_image = image.copy()
         return {"bbox" :bbox}
      else:
         if changeDetector.get_diff(image,after_image).mean() >= 0.95:
            bbox = model_flow.check(check_image,image,is_topDown= True)
            check_image = image.copy()
            if bbox:
               for (x,y,w,h) , label,state in bbox:
                  if state in ['sub','zero']:
                     ramen.loc[label,'diff'] -= 1
                  else:
                     ramen.loc[label,'diff'] += 1
               ramen.to_csv('/opt/ml/project/final-project-level3-cv-12/front/stock.csv')
               after_image = None
               print(bbox)
               return {"bbox": bbox}
   return {'bbox': bbox}
                  
   
   

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=30002)
    