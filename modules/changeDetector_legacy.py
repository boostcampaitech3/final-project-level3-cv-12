import cv2
import numpy as np
import imutils
from skimage.metrics import structural_similarity as compare_ssim



# def get_diff(gray1, gray2, diff_threshold=0.5, erosion_iter=20, dilation_iter=15):
#     _, diff = compare_ssim(gray1, gray2, full=True)
#     diff = np.where(diff < diff_threshold, 0, diff)
#     diff = (diff * 255).astype("uint8")
#     thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
#     kernel = np.ones((3,3),np.uint8)
#     erosion = cv2.erode(thresh, kernel, iterations = erosion_iter)
#     dilation = cv2.dilate(erosion, kernel, iterations = dilation_iter)

#     cnts = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     bboxes = []
#     for c in cnts:
#         (x, y, w, h) = cv2.boundingRect(c)
#         bboxes.append((x, y, w, h))

#     return bboxes


# def detectChange(before_img, after_img, diff_threshold=0.5, erosion_iter=20, dilation_iter=15):

#     before_gray = cv2.cvtColor(before_img, cv2.COLOR_RGB2GRAY)
#     after_gray = cv2.cvtColor(after_img, cv2.COLOR_RGB2GRAY)

#     bboxes = get_diff(before_gray, after_gray, diff_threshold, erosion_iter, dilation_iter)
    
#     return bboxes

def get_diff(before_img, after_img):
    before_gray = cv2.cvtColor(before_img, cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(after_img, cv2.COLOR_RGB2GRAY)
    _, diff = compare_ssim(before_gray, after_gray, full=True)
    diff = np.where(diff < 0.6, 0, diff)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations = 20)
    dilation = cv2.dilate(erosion, kernel, iterations = 15)
    return dilation


def detectChange(before_img, after_img, before_bboxes, after_bboxes, threshold=50):
    before_changed = []
    after_changed = []
    diff = get_diff(before_img, after_img)
    

    for bbox in before_bboxes:
        x, y, w, h = bbox
        score = diff[y:y+h, x:x+w].mean()
        if score > threshold:
            before_changed.append((bbox, score))
    
    for bbox in after_bboxes:
        x, y, w, h = bbox
        score = diff[y:y+h, x:x+w].mean()
        if score > threshold:
            after_changed.append((bbox, score))
    
    return before_changed, after_changed