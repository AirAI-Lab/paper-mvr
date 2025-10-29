import torch
import os
import cv2
from ultralytics import YOLO

# 加载YOLOv8模型
# model = YOLO("yolov8n.pt")
model = YOLO("yolov8n-obb.pt")
save_dir = r'C:\Users\ycc\Desktop\mvr\paper-mvr\data\GSBF_video'
save_name = 'GSBF_video'

# 配置跟踪参数
results = model.track(
    source=r"C:\Users\ycc\Desktop\mvr\paper-mvr\data\GSBF_video.mp4",
    # source=r'F:/DCIM/DJI_202510261502_002/reconstructed_video.mp4',
    project=save_dir,
    name=save_name,
    tracker="botsort.yaml",
    save=True,
    save_frames=True,
    save_txt=True,
    stream_buffer=True,
    vid_stride=1,
    conf=0.3,
    iou=0.5,
    show=True,
    show_labels=True,
    show_conf=True,
    show_boxes=True,
)