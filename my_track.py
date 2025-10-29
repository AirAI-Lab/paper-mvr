import torch
import os
import cv2  # 导入OpenCV用于保存图片
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO("best.pt")
save_dir = 'F:\DCIM\DJI_202510261502_002'
save_name = 'DJI_20251026150647_0002_S'

# 配置跟踪参数
results = model.track(
    source=r'F:\DCIM\DJI_202510261502_002\DJI_20251026152053_0004_S.MP4',
    project=save_dir,
    name=save_name,
    tracker="transformer_tracker.yaml",
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
)