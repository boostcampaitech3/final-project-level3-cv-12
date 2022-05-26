import cv2
import numpy as np
import imutils
from skimage.metrics import structural_similarity as compare_ssim



def get_diff(img1, img2, gray1, gray2):
    _, diff = compare_ssim(gray1, gray2, full=True)
    diff = np.where(diff < 0.8, 0, diff)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=10)
    
    cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    bboxes = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        bboxes.append((x, y, w, h))

    return bboxes


def detectChange(before_img, after_img):

    before_gray = cv2.cvtColor(before_img, cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(after_img, cv2.COLOR_RGB2GRAY)

    bboxes = get_diff(before_img, after_img, before_gray, after_gray)
    
    return bboxes