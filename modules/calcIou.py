def checkIou(box1, box2):
    
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_max1, x_max2 = x1+w1, x2+w2
    y_max1, y_max2 = y1+h1, y2+h2
    

    box1_area = (w1 + 1) * (h1 + 1) 
    box2_area = (w2 + 1) * (h2 + 1) 

    x1_inter = max(x1, x2) 
    y1_inter = max(y1, y2) 
    x2_inter = min(x_max1, x_max2) 
    y2_inter = min(y_max1, y_max2) 

    w_inter = max(0, x2_inter - x1_inter + 1)
    h_inter = max(0, y2_inter - y1_inter + 1)

    inter = w_inter * h_inter
    iou = inter / (box1_area + box2_area - inter)
    return iou


def checkIou2(box1, box2):
    
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_max1, x_max2 = x1+w1, x2+w2
    y_max1, y_max2 = y1+h1, y2+h2
    

    box1_area = (w1 + 1) * (h1 + 1) 
    box2_area = (w2 + 1) * (h2 + 1) 

    x1_inter = max(x1, x2) 
    y1_inter = max(y1, y2) 
    x2_inter = min(x_max1, x_max2) 
    y2_inter = min(y_max1, y_max2) 

    w_inter = max(0, x2_inter - x1_inter + 1)
    h_inter = max(0, y2_inter - y1_inter + 1)

    inter = w_inter * h_inter
    iou = inter / min(box1_area, box2_area)
    return iou


def CalIou(before_bboxes, after_bboxes, threshold=0.8):

    out = []
    after_bboxes = set(after_bboxes)
    for before_bbox in before_bboxes:
        best_bbox = None
        best_iou = threshold
        for after_bbox in after_bboxes:
            iou = checkIou(before_bbox, after_bbox) 
            if iou > best_iou:
                best_bbox = after_bbox
                best_iou = iou

        after_bboxes.discard(best_bbox)
        out.append((before_bbox, best_bbox)) # 없을 경우 (before_bbox, None)

    # before_bbox가 없는 after_bboxes
    for after_bbox in after_bboxes:
        out.append((None, after_bbox))
            
    return out


def CalIou2(before_bboxes, after_bboxes, threshold=0.8):

    out = set()
    
    _before_bboxes = set(before_bboxes) # None용
    _after_bboxes = set(after_bboxes) # None용

    before_overlaps = {}
    after_overlaps = {}


    for before_bbox in before_bboxes:
        for after_bbox in after_bboxes:
            iou = checkIou2(before_bbox, after_bbox)
            if iou > threshold:
                before_overlaps.setdefault(before_bbox, []).append(after_bbox)
                after_overlaps.setdefault(after_bbox, []).append(before_bbox)
                _before_bboxes.discard(before_bbox)
                _after_bboxes.discard(after_bbox)

    for before_bbox in _before_bboxes:
        out.add(f'({set([before_bbox])}, {{None}})')

    for after_bbox in _after_bboxes:
        out.add(f'({{None}}, {set([after_bbox])})')

    for before_bbox, before_overlap in before_overlaps.items():
        before_bboxes = {before_bbox}
        after_bboxes = set()
        for overlap in before_overlap: # before와 겹치는 것들
            after_bboxes.add(overlap)
            for after_overlap in after_overlaps[overlap]: # before_overlap과 겹치는 before들..
                before_bboxes.add(after_overlap)


        out.add(f'({before_bboxes}, {after_bboxes})')
    
    out2 = []
    for bbox in out:
        out2.append(eval(bbox))
    return out2