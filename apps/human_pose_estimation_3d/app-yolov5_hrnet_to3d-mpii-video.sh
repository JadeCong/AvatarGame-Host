#!/bin/bash
###
 # @Author: your name
 # @Date: 2020-12-04 10:14:51
 # @LastEditTime: 2021-01-05 20:11:55
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /Workflows/server/PycharmProjects/Pacific_AvatarGame_Host/apps/human_pose_estimation_3d/app-yolov5_hrnet_to3d-mpii-video.sh
### 

# launch the dnsmasq service
srv_status_dnsmasq=`service dnsmasq status | grep 'Active' | awk '{print $2}'`
if [ $srv_status_dnsmasq = "active" ]; then
    echo "The status of service dnsmasq is running and dns configuration gets ready."
elif [ $srv_status_dnsmasq = "inactive" -o $srv_status_mosquitto = "failed" ]; then
    echo "The status of service dnsmasq is dead and launch the service for dns configuration."
    service dnsmasq start
fi

# launch the mosquitto service
srv_status_mosquitto=`service mosquitto status | grep 'Active' | awk '{print $2}'`
if [ $srv_status_mosquitto = "active" ]; then
    echo "The status of service mosquitto is running and broker gets ready."
    echo "=================================================="
elif [ $srv_status_mosquitto = "inactive" -o $srv_status_mosquitto = "failed" ]; then
    echo "The status of service mosquitto is dead and launch the service for broker."
    echo "=================================================="
    service mosquitto start
fi

# launch the application for 3d human pose estimation(yolov5, hrnet and to3d) with MPII dataset(classes 0 for human class in COCO dataset for human detection)
python3 app-yolov5_hrnet_to3d-mpii-video.py \
    --config app-yolov5_hrnet_to3d-mpii-video.yaml \
    --device 0 \
    --local_rank 0 \
    --seed 26 \
    --weights $(cd $(dirname $0) && cd ../../data && pwd -P)/weights/pretrained/yolov5/pytorch/yolov5/yolov5s.pt \
    --source "/home/pacific/Documents/Work/Projects/Workflows/server/PycharmProjects/Pacific_AvatarGame_Host/resources/videos/daniel_xiaomi_1920x1080.mp4" \
    --img_size 640 \
    --conf_thres 0.8 \
    --view_img \
    --save_img \
    --save_label \
    --save_conf \
    --save_pose_2d \
    --save_pose_3d \
    --save_msg_pub \
    --classes 0 \
    --broker_address "127.0.0.1" \
    --topic_pub "/pacific/avatar/human_keypoints_3d" \
    --process_human_number 1 \
    --filter_2d_pose \
    --filters_2d_params 5 5 0.01 \
    --filter_3d_pose \
    --filters_3d_params 5 5 0.01 \
    --view_pose_3d \
    --axis_range_3d [-2,2] [-2,2] [-2,2] \
    --axis_scale_3d 0.5 0.5 0.5
