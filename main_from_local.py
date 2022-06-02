import cv2
import warnings
warnings.filterwarnings("ignore")

from modules.model import stockChecker
from modules.utils import *

stockchecker = stockChecker()
f_video = 'KakaoTalk_20220530_123105888.mp4'

lw = 5
before_frame = None

cap = cv2.VideoCapture(f_video)


if cap.isOpened():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) 
    video = cv2.VideoWriter(f'output_{f_video}', fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        is_human = stockchecker.check_human(frame)
        if is_human:
            video.write(frame)

        else:
            if type(before_frame) == type(None):
                before_frame = frame
                state_cum = set()
                video.write(frame)
            else:
                state_now = stockchecker.check(before_frame, frame)
                after_frame = frame.copy()
                state_cum = combine_state(state_cum, state_now)

                for (x, y, w, h), state in state_cum:
                    if state == "zero":
                        cv2.rectangle(after_frame, (x, y), (x+w, y+h), (0, 0, 255), lw) #빨강
                    if state == 'new': 
                        cv2.rectangle(after_frame, (x, y), (x+w, y+h), (255, 0, 0), lw) #파랑
                    if state == 'sub':
                        cv2.rectangle(after_frame, (x, y), (x+w, y+h), (147, 20, 255), lw) # 자주
                    if state == 'add':
                        cv2.rectangle(after_frame, (x, y), (x+w, y+h), (255, 191, 0), lw) # 하늘
                video.write(after_frame)

                before_frame = frame
