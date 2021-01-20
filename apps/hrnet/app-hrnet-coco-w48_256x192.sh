#!/bin/bash

# launch the application for human keypoints estimation(hrnet) with coco dataset
python3 app-hrnet-coco.py \
    --cfg app-hrnet-coco-w48_256x192.yaml \
    --videoFile 0 \
    --writeBoxFrames \
    --outputDir output \
    TEST.MODEL_FILE ../../data/weights/pretrained/hrnet/pytorch/pose_coco/pose_hrnet_w48_256x192.pth
