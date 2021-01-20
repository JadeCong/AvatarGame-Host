###
 # @Author: your name
 # @Date: 2020-12-31 15:46:57
 # @LastEditTime: 2021-01-04 17:43:18
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /Workflows/server/PycharmProjects/Pacific_AvatarGame_Host/apps/drive_avatar_by_pose/app-publish_human_pose-30fps.sh
### 

python3 app-publish_human_pose-30fps.py \
    --broker_address "127.0.0.1" \
    --topic_pub "/pacific/avatar/human_keypoints_3d" \
    --pub_fps 3.5 \
    --data_path "../human_pose_estimation_3d/runs/detect/exp1/messages/example_640x480_msgs_pub.txt"
