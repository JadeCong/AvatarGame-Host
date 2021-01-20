#!/bin/bash

# launch the application for human keypoints estimation(hrnet) with mpii dataset
python3 app-hrnet-mpii.py \
    --cfg app-hrnet-mpii-w32_256x256.yaml \
    --videoFile 0 \
    --writeBoxFrames \
    --outputDir output \
    TEST.MODEL_FILE ../../data/weights/pretrained/hrnet/pytorch/pose_mpii/pose_hrnet_w32_256x256.pth
