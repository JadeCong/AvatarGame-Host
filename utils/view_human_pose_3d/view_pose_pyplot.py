'''
Author: your name
Date: 2020-12-21 14:27:25
LastEditTime: 2020-12-23 10:28:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Workflows/server/PycharmProjects/Pacific_AvatarGame_Host/utils/view_human_pose_3d/view_pose_pyplot.py
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time


class HumanPose3DViewer():
    def __init__(self, human_mode):
        # plt.ion()  # for interactive plot
        self.fig = plt.figure("3D Human Pose Estimation", figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(20, 45)  # unity view for human pose
        
        if human_mode == "pacific" or human_mode == "PACIFIC":
            self.data_order = [[6,2,1,0], [6,3,4,5], [6,7,8,9,10], [8,13,12,11], [8,14,15,16]]
        elif human_mode == "mpii" or human_mode == "MPII":
            self.data_order = [[6,2,1,0], [6,3,4,5], [6,7,8,9], [8,12,11,10], [8,13,14,15]]
        self.data_color = ['red', 'blue', 'green', 'red', 'blue']
        
        self.x_data = []
        self.y_data = []
        self.z_data = []
    
    def configureViewAngle(self):
        self.elev = self.ax.elev
        self.azim = self.ax.azim
        
        self.ax.view_init(self.elev, self.azim)
    
    def configureAxis(self, axis_range, axis_scale):
        self.ax.set_title("3D Human Pose Estimation")
        
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_zlabel('Z-axis')
        self.ax.xaxis.label.set_color('red')
        self.ax.yaxis.label.set_color('green')
        self.ax.zaxis.label.set_color('blue')
        
        # TODO: Set the axis arrow type and the axis color
        # self.ax.axis['x'].set_axisline_style('-|>', size=3)
        # self.ax.axis['y'].set_axisline_style('-|>', size=3)
        # self.ax.axis['z'].set_axisline_style('-|>', size=3)
        self.ax.tick_params(axis='x', colors='red')
        self.ax.tick_params(axis='y', colors='green')
        self.ax.tick_params(axis='z', colors='blue')
        
        self.ax.set_xlim3d([axis_range[0][0], axis_range[0][1]])
        self.ax.set_ylim3d([axis_range[1][0], axis_range[1][1]])
        self.ax.set_zlim3d([axis_range[2][0] + axis_range[2][1], axis_range[2][1] - axis_range[2][0]])
        self.ax.set_xticks(np.arange(axis_range[0][0], axis_range[0][1], axis_scale[0]))
        self.ax.set_yticks(np.arange(axis_range[1][0], axis_range[1][1], axis_scale[1]))
        self.ax.set_zticks(np.arange(axis_range[2][0] + axis_range[2][1], axis_range[2][1] - axis_range[2][0], axis_scale[2]))
    
    def viewPose(self, human_kps_3d, axis_range, axis_scale):
        data_shape = human_kps_3d.shape
        human_data = human_kps_3d.tolist()
        
        for human in range(data_shape[0]):
            for branch in range(len(self.data_order)):
                for idx in range(len(self.data_order[branch])):
                    self.x_data.append(human_data[human][self.data_order[branch][idx]][0])
                    self.y_data.append(human_data[human][self.data_order[branch][idx]][1])
                    self.z_data.append(human_data[human][self.data_order[branch][idx]][2])  # set the axises for current(0,1,2)/unity(0,2,1) order
                    
                    self.ax.text(human_data[human][self.data_order[branch][idx]][0], human_data[human][self.data_order[branch][idx]][1], \
                        human_data[human][self.data_order[branch][idx]][2], self.data_order[branch][idx], size='medium')
                
                self.configureViewAngle()
                self.configureAxis(axis_range, axis_scale)
                self.ax.scatter(self.x_data, self.y_data, self.z_data, color=self.data_color[branch], marker='o', s=10)
                self.ax.plot(self.x_data, self.y_data, self.z_data, color=self.data_color[branch], linewidth=1.5)
                
                self.x_data.clear()
                self.y_data.clear()
                self.z_data.clear()
        
        plt.pause(0.0005)
        plt.cla()
    
    def closeAll(self):
        plt.clf()
        plt.close()
