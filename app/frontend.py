import streamlit as st

import io
import os
import yaml
import torch
import cv2
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation,rc

# from predict import load_model, get_prediction

from confirm_button_hack import cache_on_button_press

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


root_password = 'password'
fake_list = {
    '신라면':3*4,
    '정비빔':3*1,
    '백비빔':3*2,
    '짜파게티':3*3
}

@st.cache(allow_output_mutation=True)
def get_cap(uploaded_file):
    return cv2.VideoCapture(str(uploaded_file))


def main():
    
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png","mp4","mpeg"])

    temporary_location = False
    
    if uploaded_file is not None:
        video_file  = uploaded_file.getvalue()
        video_bytes = video_file

        st.video(video_bytes)

        g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
        temporary_location = "testout_simple.mp4"

        with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
            out.write(g.read())  ## Read bytes into file

        # close file
        out.close()
        
        file_path = '/opt/ml/Boostcamp-AI-Tech-Product-Serving/part2/02-streamlit/testout_simple.mp4'
        if os.path.isfile(file_path):
            cap = cv2.VideoCapture(file_path)
        else:
            print('파일이 존재하지 않습니다.')
        
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #capture후 prediction

        frame_size =(frameWidth,frameHeight)
        print(f'frame_size={frame_size}')

        num = 0
        while num < 5:
            ret,frame = cap.read()
            if not(ret): #프레임 정보를 정상적으로 읽지 못하면
                break
            # print(type(new_frame))
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            
            # print(type(frame))
            # print(frame.shape)
            img_bytes = cv2.imencode('.jpg',frame)[1].tobytes()
            data = io.BytesIO(img_bytes)
            response = requests.post("http://localhost:8001/order", data=data)    
    
            print(response.json())

            key = cv2.waitKey(33)
            if key == 27:
                break
            num+=1
        if cap.isOpened():
            cap.release()

      
        # st.write(response.json())



@cache_on_button_press('Authenticate')
def authenticate(password) ->bool:
    print(type(password))
    return password == root_password


password = st.text_input('password', type="password")

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error('The password is invalid.')