# YOLOv5 Real-Time Face and Object Detection

This project implements real-time object detection using a webcam feed powered by **YOLOv5**. It supports face detection and identifies all objects listed in the `coco.names` dataset.

## 🚀 Features

- Real-time object detection using webcam or USB camera
- Face detection (via bounding boxes or as a custom class)
- Detects all 80 COCO dataset classes (e.g., person, car, dog, etc.)
- Customizable detection threshold and display settings
- Runs on CPU or GPU

## 🧠 Model Used

- **YOLOv5** (You Only Look Once) by Ultralytics
- Pretrained weights on COCO dataset (`yolov5s.pt`, `yolov5m.pt`, etc.)

## 📦 coco.names – COCO Dataset Classes

The model detects the following 80 object classes, including:

- Person
- Bicycle
- Car
- Dog
- Cat
- Chair
- Cell phone
- TV
- ...and more

(Full list available in `coco.names` or at https://github.com/ultralytics/yolov5/blob/master/data/coco.names)

## 👤 Face Detection Note

YOLOv5 does not natively include "face" as a COCO class. For specific face detection:

- Train YOLOv5 on a face dataset (e.g., WIDER FACE)
- Or use OpenCV’s Haar cascades or DNN models alongside YOLO

## 🙌 Acknowledgements

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [COCO Dataset](https://cocodataset.org/)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
