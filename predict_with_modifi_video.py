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
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# Tạo video output với thông số tương tự như video input
# output_video_path = 'output_video.mp4'
# out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

names = model.names
list = ["car", "truck", "bus"]
area_road = find_area_road(mask)

start = time.time()
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    # frame = image_process(frame, mask)
    results = model.predict(source=frame, save=True, boxes = False, conf = 0.15)  # predict() returns a named tuple
    # print(results[0].boxes.cls)
    # Sum_area = 0
    # num_car = 0
    # try:
    #     for i in range(len(results[0].masks)):
    #         if names[int(results[0].boxes.cls[i])] in list:
    #             Sum_area += np.count_nonzero(results[0].masks.data[i])
    #             num_car += 1
    # except:
    #     pass

    # print("total: {} pixel".format(Sum_area))
    # print("Số xe chiếm: {} % lòng đường".format(np.round(Sum_area/area_road*100, 2)))
    # print('Số lượng xe: {}'.format(num_car))
    cv2.imshow('frame', cv2.imread(f'{results[0].save_dir}\image0.jpg'))
    # out.write(cv2.imread(f'{results[0].save_dir}\image0.jpg'))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print('time: ', time.time() - start)

cap.release()
# out.release()
cv2.destroyAllWindows()



