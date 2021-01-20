'''
Author: your name
Date: 2020-12-21 16:57:47
LastEditTime: 2020-12-25 19:29:48
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Workflows/server/PycharmProjects/Pacific_AvatarGame_Host/utils/view_human_pose_3d/test.py
'''
from view_pose_pyplot import HumanPose3DViewer
import numpy as np
from numpy import random


# data_viewer = HumanPose3DViewer("pacific")

# while True:
#     data = random.randint(1, 10, (17*3*2,)) / 5
#     data = data.reshape(2,17,3)
#     data_viewer.viewPose(data, [[1,5], [1,5], [1, 5]], [1, 1, 1])



import argparse

parser = argparse.ArgumentParser(description='Process the argument inputs...')
parser.add_argument('--axis_range_3d', action='store', nargs=3, type=list, metavar=('x_range', 'y_range', 'z_range'), default=[[-1,1], [-1,1], [-1,1]], help="Configure the axis range for 3D human pose viewer")
# parser.add_argument('--axis_scale_3d', action='store', nargs=3, type=float, metavar=('x_scale', 'y_scale', 'z_scale'), default=[0.2, 0.2, 0.2], help="Configure the axis scale for 3D human pose viewer")
args = parser.parse_args()

print(args.axis_range_3d)
# print(args.axis_scale_3d)
