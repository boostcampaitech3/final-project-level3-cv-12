def get_bbox_info(x, y, w, h):
    x_center = x + w//2
    y_center = y + h//2
    x_min, y_min, x_max, y_max = x, y, x+w, y+h
    return x_min, y_min, x_max, y_max, x_center, y_center


def get_side_bbox(bbox, bboxes):
    x_min, y_min, x_max, y_max, x_center, y_center = get_bbox_info(*bbox)

    min_left_distance = 100000
    min_right_distance = 100000

    for bbox2 in bboxes:
        bbox2_info = get_bbox_info(*bbox2)
        x_min2, y_min2, x_max2, y_max2, x_center2, y_center2 = bbox2_info
        cond1 = y_min < y_center2 < y_max  # 범위 체크 1
        cond2 = y_min2 < y_center < y_max2 # 범위 체크 2
        if cond1 and cond2:
            distance = abs(x_center2 - x_center)

            if x_center2 < x_center: #left
                if distance < min_left_distance:
                    min_left_distance = distance
                    min_left = bbox2_info

            elif x_center2 > x_center: #right
                if distance < min_right_distance:
                    min_right_distance = distance
                    min_right = bbox2_info

    return min_left, min_right

def get_side_bbox2(bbox, bboxes):

    x_min, y_min, x_max, y_max, x_center, y_center = 1e8, 1e8, 0, 0, 0, 0
    for b in bbox:
        _x_min, _y_min, _x_max, _y_max, _x_center, _y_center = get_bbox_info(*b)
        x_min = min(x_min, _x_min)
        y_min = min(y_min, _y_min)
        x_max = max(x_max, _x_max)
        y_max = max(y_max, _y_max)
        x_center += _x_center
        y_center += _y_center
    x_center /= len(bbox)
    y_center /= len(bbox)

    min_left_distance = 100000
    min_right_distance = 100000
    
    min_left = None
    min_right = None
    for bbox2 in bboxes:
        bbox2_info = get_bbox_info(*bbox2)
        x_min2, y_min2, x_max2, y_max2, x_center2, y_center2 = bbox2_info
        cond1 = y_min < y_center2 < y_max  # 범위 체크 1
        cond2 = y_min2 < y_center < y_max2 # 범위 체크 2
        if cond1 and cond2:
            distance = abs(x_center2 - x_center)

            if x_center2 < x_center: #left
                if distance < min_left_distance:
                    min_left_distance = distance
                    min_left = bbox2_info

            elif x_center2 > x_center: #right
                if distance < min_right_distance:
                    min_right_distance = distance
                    min_right = bbox2_info

    return min_left, min_right

def get_slope(bbox, bboxes):
    x_center, y_center = get_bbox_info(*bbox)[-2:]
    min_left, min_right = get_side_bbox(bbox, bboxes)
    left_x, left_y = min_left[-2:]
    right_x, right_y = min_right[-2:]
    left_slope =  -round((left_y - y_center) / (left_x - x_center), 2)
    right_slope =  -round((right_y - y_center) / (right_x - x_center), 2)
    
    return left_slope, right_slope

def get_slope2(bbox, bboxes):
    x_center, y_center = 0, 0
    for b in bbox:
        _x_center, _y_center = get_bbox_info(*b)[-2:]
        x_center += _x_center
        y_center += _y_center
    
    x_center /= len(bbox)
    y_center /= len(bbox)

    min_left, min_right = get_side_bbox2(bbox, bboxes)

    left_slope, right_slope = None, None
    if min_left != None:
        left_x, left_y = min_left[-2:]
        left_slope =  -round((left_y - y_center) / (left_x - x_center), 2)

    if min_right != None: 
        right_x, right_y = min_right[-2:] 
        right_slope =  -round((right_y - y_center) / (right_x - x_center), 2)
    
    return left_slope, right_slope

def check_status(before_bbox, after_bbox, before_bboxes, after_bboxes):
    
    before_left_slope, before_right_slope = get_slope(before_bbox, before_bboxes)
    after_left_slope, after_right_slope = get_slope(after_bbox, after_bboxes)

    d_left = (after_left_slope - before_left_slope) 
    d_right = (after_right_slope - before_right_slope)
    
    prod = d_left - d_right
    return prod

def check_status2(before_bbox, after_bbox, before_bboxes, after_bboxes):
    
    before_left_slope, before_right_slope = get_slope2(before_bbox, before_bboxes)
    after_left_slope, after_right_slope = get_slope2(after_bbox, after_bboxes)

    d_left, d_right = 0, 0
    if  after_left_slope != None and before_left_slope != None:
        d_left = (after_left_slope - before_left_slope) 
    if after_right_slope != None and before_right_slope != None:
        d_right = (after_right_slope - before_right_slope)
    
    prod = d_left - d_right
    return prod