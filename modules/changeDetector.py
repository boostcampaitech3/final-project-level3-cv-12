import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim


def get_diff(before_img, after_img):
    before_gray = cv2.cvtColor(before_img, cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(after_img, cv2.COLOR_RGB2GRAY)
    _, diff = compare_ssim(before_gray, after_gray, full=True)
    diff = np.where(diff < 0.6, 0, diff)
    diff = (diff * 255).astype("uint8")

    if diff.std() < 10:
        diff = 255-diff
        return diff

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations = 20)
    dilation = cv2.dilate(erosion, kernel, iterations = 15)
    return dilation


def detectChange(before_img, after_img, before_bboxes, after_bboxes, threshold=100):
    before_changed = []
    before_scores = []
    after_changed = []
    after_scores = []

    diff = get_diff(before_img, after_img)
    
    for bbox in before_bboxes:
        x, y, w, h = bbox
        score = diff[y:y+h, x:x+w].mean()
        if score > threshold:
            before_changed.append(bbox)
            before_scores.append(score)

    for bbox in after_bboxes:
        x, y, w, h = bbox
        score = diff[y:y+h, x:x+w].mean()
        if score > threshold:
            after_changed.append(bbox)
            after_scores.append(score)

    # 변화된 bbox 제거
    for discard in before_changed:
        before_bboxes.discard(discard)

    for discard in after_changed:
        after_bboxes.discard(discard)

    return before_changed, before_scores, after_changed, after_scores