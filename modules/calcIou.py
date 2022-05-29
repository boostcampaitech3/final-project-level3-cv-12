def checkIou(box1,box2):
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

def CalIou(all_box,change_box,size):
    new_all_box , new_change_box = [],[]

    for i in all_box:
        xc,yc,w,h = float(i[1]) * size , float(i[2]) * size ,float(i[3]) * size , float(i[4]) * size
        xmin = xc- w /2
        ymin = yc - h /2
        xmax = w + xmin
        ymax = h + ymin
        xmin , ymin , xmax, ymax = round(xmin,6),round(ymin,6),round(xmax,6),round(ymax,6)
        new_all_box.append([i[0],xmin,ymin,xmax,ymax,i[5]])

    for i in change_box:
        xc,yc,w,h = float(i[1]) * size , float(i[2]) * size ,float(i[3]) * size , float(i[4]) * size
        xmin = xc- w /2
        ymin = yc - h /2
        xmax = w + xmin
        ymax = h + ymin
        xmin , ymin , xmax, ymax = round(xmin,6),round(ymin,6),round(xmax,6),round(ymax,6)
        new_change_box.append([i[0],xmin,ymin,xmax,ymax,i[5]])


    out = []

    for change_check in new_change_box:
        for all_check in new_all_box:
            c = [change_check[1],change_check[2],change_check[3],change_check[4]]
            a = [all_check[1],all_check[2],all_check[3],all_check[4]]
            if checkIou(c,a) > 0.8:
                out.append(all_check)
            
    
    return out