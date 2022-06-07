
import streamlit as st
import requests
import av
import queue
import cv2
import time
import numpy as np
from typing import List, NamedTuple
from streamlit.type_util import convert_anything_to_df
from streamlit_webrtc import (VideoProcessorBase, WebRtcMode, webrtc_streamer, RTCConfiguration)

def ramen_app():
    class Video(VideoProcessorBase):
        '''
        Video process
        '''
        
        def __init__(self) -> None:
            self.result_queue = queue.Queue()
            self.frame_cnt = 0
            self.response = 0

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame: 
            
            image = frame.to_ndarray(format="bgr24")

            if self.frame_cnt % 3 == 0:
                
                file = encode_image(image)
                
                self.response = requests.post("http://118.67.132.27:30001/", files=file) # Add your detection server address
                
            bbox = self.response.json()['bbox']
            if bbox is not None:
                image = annotate(image,bbox)                 

            return av.VideoFrame.from_ndarray(image, format="bgr24")
    frame_rate = 15
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV, 
        # rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=Video,
        media_stream_constraints={"video": {"frameRate": {"ideal": frame_rate}, "width": 2048, "height": 2048}, "audio": False}, 
        async_processing=True,
    )

    

def encode_image(image):
    '''
    Encoding image to bytes
    '''  
    _, img_encoded = cv2.imencode('.jpg', image)

    return {'image':img_encoded.tobytes()}




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

    
def main():
    ramen_app()
    

if __name__ == "__main__":
    main()