from turtle import dot
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
import cv2

class pixel_simularity:
    def __init__(self,before_changed, after_changed,before_img,after_img):
        self.before_changed =  before_changed 
        self.after_changed = after_changed
        self.before_img = before_img
        self.after_img = after_img

    def dot_product(self,A,B):
        color = ('r','g','b')
        cosin_sim = []
        for i, c in enumerate(color):
            res = np.dot(np.squeeze(np.asarray(A[i])),np.transpose(np.squeeze(np.asarray(B[i])))) / (norm(np.squeeze(np.asarray(A[i]))) * norm(np.squeeze(np.asarray(B[i]))))
            cosin_sim.append(res)
            print(c , ':' , res)
        mean_sim = np.asarray(cosin_sim) @ np.array([1,1,1]) / 3
        
        return mean_sim

    def cal_sim(self):
        a = self.before_changed[0]
        b = self.after_changed[0]
        #x,y,w,h
        crop_a = self.before_img[a[1]:a[1]+a[3],a[0]:a[0]+a[2]]
        crop_b = self.after_img[b[1]:b[1]+b[3],b[0]:b[0]+b[2]]

        rgb_hist = []
        temp = []
        color =('b','g','r')

        for i,col in enumerate(color):
            histr = cv2.calcHist([crop_a],[i],None,[256],[0,256])
            temp.append(histr)
        #     plt.plot(histr,color=col)
        #     plt.xlim([0,256])
        # plt.show()
        rgb_hist.append(temp)

        temp = []
        for i,col in enumerate(color):
            histr = cv2.calcHist([crop_b],[i],None,[256],[0,256])
            temp.append(histr)
        #     plt.plot(histr,color=col)
        #     plt.xlim([0,256])
        # plt.show()
        rgb_hist.append(temp)
        if self.dot_product(rgb_hist[0],rgb_hist[1]) >= 0.9:
            return True
        else:
            return False
