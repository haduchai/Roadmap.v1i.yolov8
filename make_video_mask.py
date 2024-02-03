import gradio as gr
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from ultralytics import YOLO
import shutil
import time
# model
model = YOLO('yolov8s-seg.pt')  # set model

# tìm diện tích đường bằng cách tìm contours lớn nhất
def find_area_road(image):
    area = np.count_nonzero(image)
    return area//3

def image_process(image, mask):
    img_mask = cv2.bitwise_and(image, mask)
    return img_mask

# load video
cap = cv2.VideoCapture('video/video1.mp4')
mask = cv2.imread('video/video1_mask.jpg')

# Lấy thông tin về video (chiều rộng, chiều cao, số frame mỗi giây, ...)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Tạo video output với thông số tương tự như video input
output_video_path = 'output_video.mp4'
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

start = time.time()
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    frame = image_process(frame, mask)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print('time: ', time.time() - start)

cap.release()
# out.release()
cv2.destroyAllWindows()



