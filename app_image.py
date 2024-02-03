import gradio as gr
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from ultralytics import YOLO
import shutil

# config theme
theme = gr.themes.Default(primary_hue="blue").set(
    background_fill_primary="#cccc",
    button_secondary_background_fill="#00ced1ff",
    input_background_fill="#b8b8b8ff",
    button_large_text_size="40px",
    button_large_padding="20px 20px",
    input_padding= "20px 20px", 
    block_title_text_size="30px"  ,
    block_title_text_color= "black",
    block_background_fill="#ccc",
    input_text_size="30px"
)

# tìm diện tích đường bằng cách tìm contours lớn nhất
def find_area_road(image):
    area = np.count_nonzero(image)
    return area//3

# tìm segment
def image_segment(image_file):
    base_path = 'image/'
    path_image = os.path.basename(image_file.name)
    shutil.copyfile(image_file.name, path_image)
    image = cv2.imread(base_path + path_image)
    path_mask = path_image.split('.')[0] + '_mask.jpg'
    mask = cv2.imread(base_path + path_mask)

    # change rgb to bgr
    # image = image[:,:,::-1]

    img_mask = cv2.bitwise_and(image, mask)
    mask.resize((544, 640))
    area_road = find_area_road(mask)

    model = YOLO('yolov8s-seg.pt')  # set model
    results = model.predict(source=img_mask, save=True, boxes = False, conf = 0.15)  # predict() returns a named tuple
    names = model.names
    list = ["car", "truck", "bus"]
    Sum_area = 0
    num_car = 0
    # print(results[0].boxes.cls)
    for i in range(len(results[0].masks)):
        if names[int(results[0].boxes.cls[i])] in list:
            Sum_area += np.count_nonzero(results[0].masks.data[i])
            num_car += 1

    print("total: {} pixel".format(Sum_area))
    print("Số xe chiếm: {} % lòng đường".format(np.round(Sum_area/area_road*100, 2)))
    print('Số lượng xe: {}'.format(num_car))
    return image[:,:,::-1], gr.Image(f'{results[0].save_dir}/image0.jpg', width=800), num_car, str(np.round(Sum_area/area_road*100, 2)) + ' %'

with gr.Blocks(theme=theme) as app:
    with gr.Row():
        file_image = gr.File()

    btn = gr.Button("Submit")
    with gr.Row():
        img = gr.Image()
        img_output = gr.Image()
        with gr.Column():
            num_car = gr.Textbox(label="Số lượng xe: ", elem_classes="feedback")
            percent = gr.Textbox(label="Phần trăm chiếm: ", elem_classes="feedback")
    btn.click(image_segment, inputs=[file_image], outputs=[img, img_output, num_car, percent])

if __name__ == "__main__":
    app.launch()