from .Human_detect import Humandetect
from .changeDetector import detectChange, get_diff
from .ramen_detect import Ramen
from .calcIou import CalIou, checkIou, CalIou2, checkIou2
from .classification import Classifier
from .outDetection import get_bbox_info, get_side_bbox, get_slope, check_status, check_status2


class stockChecker():
    
    def __init__(self):
        self.humanDetector = Humandetect()
        self.ramenDetector = Ramen()
        self.classifier = Classifier()
    
    def check_human(self, img):
        ret = self.humanDetector.detect(img)
        return ret

        
    def check(self, before_img, after_img, is_topDown = True):
        before_bboxes = self.ramenDetector.ramen_detect(before_img)
        after_bboxes = self.ramenDetector.ramen_detect(after_img) 
        before_changed, before_scores, after_changed, after_scores = detectChange(before_img, after_img, before_bboxes, after_bboxes, threshold=50) 
        iou_bboxes = CalIou2(before_changed, after_changed, threshold=0.4)
        # print(iou_bboxes)
        
        out = set()
        for before_bbox, after_bbox in iou_bboxes:
            if None in before_bbox:
                out.add((after_bbox.pop(), 'new'))#하나도 없다가 생긴경우
            elif None in after_bbox:
                out.add((before_bbox.pop(), 'zero'))#하나도 없는경우
            else:
                status = check_status2(before_bbox, after_bbox, before_bboxes, after_bboxes) # -1: 빠짐, 0: 유지, 1: 추가
                status = -status if is_topDown else status
                if status < 0:
                    for bbox in before_bbox:
                        out.add((bbox, 'sub'))# 하나 빠지고 상품이 있는경우
                elif status > 0:
                    for bbox in after_bbox:
                        out.add((bbox, 'add'))# 상품이 있었는데 하나 추가된경우
        return out