#!/bin/bash

# launch the application for human keypoints estimation(resnet) with coco dataset
python3 app-resnet-coco.py \
    --cfg app-resnet-coco-101_256x192.yaml \
    --videoFile 0 \
    --writeBoxFrames \
    --outputDir output \
    TEST.MODEL_FILE ../../data/weights/pretrained/hrnet/pytorch/pose_coco/pose_resnet_101_256x192.pth
