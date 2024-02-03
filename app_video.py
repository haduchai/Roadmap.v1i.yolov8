import gradio as gr
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from ultralytics import YOLO
import shutil
import torch

# load model
model = YOLO('yolov8s-seg.pt')  # set model
names = model.names
list = ["car", "truck", "bus"]

# cải thiện tốc độ video, cứ threshold_frame thì mới predict
threshold_frame = 0

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
    return area

def image_process(image, mask):
        img_mask = cv2.bitwise_and(image, mask)
        return img_mask

# tìm segment
def image_segment(image_file):
    base_path = 'video/'
    path_video = os.path.basename(image_file.name)
    shutil.copyfile(image_file.name, path_video)

    path_mask = path_video.split('.')[0] + '_mask.jpg'
    mask = cv2.imread(base_path + path_mask)
    mask_ = mask.copy()

    cap = cv2.VideoCapture(path_video)

    Sum_area = 0
    num_car = 0
    frame_count = 0
    ret, frame = cap.read()
    i = 0
    while ret:
        frame_process = image_process(frame, mask)
        if frame_count < threshold_frame:
            frame_count += 1
        else:
            Sum_area = 0
            num_car = 0
            frame_count = 0
            results = model.predict(source=frame_process, save=True, boxes = False, conf = 0.15)  # predict() returns a named tuple
            try:
                masks = results[0].masks.data
                boxes = results[0].boxes.data

                # lấy index class của yolo
                clss = boxes[:, 5]
                # lọc index của car, truck, bus
                indices_car = torch.where(clss == 2)
                indices_truck = torch.where(clss == 7)
                indices_bus = torch.where(clss == 5)

                # lấy mask của car, truck, bus
                car_masks = masks[indices_car]
                truck_masks = masks[indices_truck]
                bus_masks = masks[indices_bus]

                # chuyển thành ảnh binary
                car_mask_ = torch.any(car_masks, dim=0).int() * 255
                truck_mask_ = torch.any(truck_masks, dim=0).int() * 255
                bus_mask_ = torch.any(bus_masks, dim=0).int() * 255

                # cộng ảnh
                mask_total = car_mask_ + truck_mask_ + bus_mask_ 
                cv2.imwrite(f'mask_total/image{i}.jpg', mask_total.cpu().numpy())       
                i += 1
                # tính lưu lượng 
                Sum_area += np.count_nonzero(mask_total)
                num_car = len(clss)
                # resize mask
                mask_.resize(results[0].masks.data[0].shape)
            except:
                pass
        area_road = find_area_road(mask_)
        # yield frame[:,:,::-1], frame_process[:,:,::-1], num_car, str(np.round(Sum_area/area_road*100, 2)) + ' %'
        yield frame[:,:,::-1], gr.Image(f'{results[0].save_dir}/image0.jpg', width=500), num_car, str(np.round(Sum_area/area_road*100, 2)) + ' %'
        ret, frame = cap.read()
    # yield frame[:,:,::-1], frame_process[:,:,::-1], num_car, str(np.round(Sum_area/area_road*100, 2)) + ' %'
    yield frame[:, :, ::-1], gr.Image(f'{results[0].save_dir}/image0.jpg', width=500), num_car, str(np.round(Sum_area/area_road*100, 2)) + ' %'

# gradio interface
with gr.Blocks(theme=theme) as app:
    with gr.Row():
        file_input = gr.File()

    btn = gr.Button("Submit")
    with gr.Row():
        video_input = gr.Image()
        video_output = gr.Image()
        with gr.Column():
            num_car = gr.Textbox(label="Số lượng xe: ")
            percent = gr.Textbox(label="Phần trăm chiếm: ")
    btn.click(image_segment, inputs=[file_input], outputs=[video_input, video_output, num_car, percent])

if __name__ == "__main__":
    app.queue()
    app.launch()