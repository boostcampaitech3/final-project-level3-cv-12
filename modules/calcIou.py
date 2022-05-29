def checkIou(box1,box2):
    
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_max1, x_max2 = x1+w1, x2+w2
    y_max1, y_max2 = y1+h1, y2+h2

    box1_area = (w1 + 1) * (h1 + 1) # box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)# (w + 1) * (h + 1)
    box2_area = (w2 + 1) * (h2 + 1) # box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)# (w + 1) * (h + 1)

    x1_inter = max(x1, x2) # x1 = max(box1[0], box2[0])
    y1_inter = max(y1, y2) # y1 = max(box1[1], box2[1])
    x2_inter = min(x_max1, x_max2) # x2 = min(box1[2], box2[2])
    y2_inter = min(y_max1, y_max2) # y2 = min(box1[3], box2[3])

    
    w_inter = max(0, x2_inter - x1_inter + 1)
    h_inter = max(0, y2_inter - y1_inter + 1)

    inter = w_inter * h_inter
    iou = inter / (box1_area + box2_area - inter)
    return iou


def CalIou(all_box, change_box, threshold=0.8):

    out = []
    for change_check in change_box:
        for all_check in all_box:
            if checkIou(change_check, all_check) > threshold:
                out.append(all_check)
            
    
    return out