import cv2
import pandas as pd

import argparse

import modules.model as model
import modules.changeDetector as changeDetector

model_flow = model.stockChecker()

def annotate(img,bbox):
    lw = 5
    for (x,y,w,h),label,state in bbox:
        if state == "zero":
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), lw) #빨강
        if state == 'new': 
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), lw) #파랑
        if state == 'sub':
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (147, 20, 255), lw) # 자주
        if state == 'add':
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 191, 0), lw) # 하늘

    return img


def detect(opt):
    cap = cv2.VideoCapture(opt.video_name)
    bbox = []
    ramen = {}
    first = True
    human = False
    cnt = 0
    flag = False
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter(opt.output_name,fourcc,cap.get(cv2.CAP_PROP_FPS),(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    for cnt in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        if cnt % 5 != 0:
            ret ,image = cap.read()
            
            if not ret:
                break
            if bbox:
                image = annotate(image,bbox)
            out.write(image)
            continue
        ret , image = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
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
            # ramen.to_csv('front/stock.csv')
            
            
        if not human:
            human = model_flow.check_human(image)  # 
        if human and not model_flow.check_human(image):
            _, next_image = cap.read()
            flag = True
            if changeDetector.get_diff(image,next_image).mean() >= 0.95:
                bbox= model_flow.check(check_image,image,is_topDown=opt.topdown)
                check_image = image.copy()
                if bbox:
                    for (x,y,w,h) , label,state in bbox:
                        if label not in ramen.index:
                            continue
                        if state in ['sub','zero']:
                            ramen.loc[label,'diff'] -= 1
                        else:
                            ramen.loc[label,'diff'] += 1
                    image = annotate(image,bbox)
                    ramen.to_csv('front/stock.csv')
                human = False
                
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        out.write(image)
        if flag:
            out.write(next_image)
            flag = False
    out.release()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name',type = str)
    parser.add_argument('--output_name',type = str,default = "./output_test_stream_check.mp4")
    parser.add_argument('--topdown',type=bool)
    opt = parser.parse_args()
    detect(opt)