#!/bin/bash

# launch the application for human detection(yolov5) with COCO dataset
python3 app-yolov5-coco.py \
    --classes 0 \
    --conf-thres 0.8 \
    --weights $(cd $(dirname $0) && cd ../../data && pwd -P)/weights/pretrained/yolov5/pytorch/yolov5/yolov5l.pt \
    --source 0 \
    --device 0  # classes 0 for human class in COCO dataset
