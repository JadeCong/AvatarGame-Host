#!/bin/bash

# launch the application for human keypoints estimation(resnet) with mpii dataset
python3 app-resnet-mpii.py \
    --cfg app-resnet-mpii-50_256x256.yaml \
    --videoFile 0 \
    --writeBoxFrames \
    --outputDir output \
    TEST.MODEL_FILE ../../data/weights/pretrained/hrnet/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar
