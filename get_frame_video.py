import gradio as gr
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from ultralytics import YOLO

cap = cv2.VideoCapture('video/video6.mp4')

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('frames/frame6.jpg', frame)
    break

cap.release()
cv2.destroyAllWindows()