import os
import numpy as np
import cv2

def read_txt(txt_file):
    key_points = np.zeros((17,2),np.float)
    with open(txt_file) as f:
        line = f.readlines()
    for i in range(17):
        key_points[i, 0] = float(line[0].split(',')[2*i])
        key_points[i, 1] = float(line[0].split(',')[2 * i + 1])
    return key_points


def gen_kepoint_file(txt_total_path,frams_num):
    key_points = np.zeros((frams_num,17,2),np.float)
    for i in range(frams_num):
        per_frame_txt = txt_total_path + str(i).zfill(4) + '.txt'
        per_frame_kepoints = read_txt(per_frame_txt)
        key_points[i,:,:] = per_frame_kepoints
    return key_points

#
# txt_total_path = './data/res/'
# key_points = gen_kepoint_file(txt_total_path,7002)
# # key_points = np.load('./girl.npy')
#
#
# cap = cv2.VideoCapture('./video/girl.mp4')
# print(cap.isOpened())
# success = True
#
# fra = 0
# while success:
#
#     success, frame = cap.read()
#     color1 = [0, 0, 200]
#     color2 = [0, 255, 0]
#     color3 = [255, 0, 0]
#     color4 = [10, 100, 255]
#     color5 = [100, 0, 200]
#
#     skeleton = [[0, 1, color1], [1, 2, color1], [2, 3, color3], [0, 4, color3], [4, 5, color5],
#                 [5, 6, color1],
#                 [0, 7, color3], [7, 8, color5], [8, 9, color2], [9, 10, color4], [8, 14, color2], [8, 11, color4],
#                 [11, 12, color5],
#                 [12, 13, color5], [14, 15, color5], [15, 16, color5]]
#
#     for ske in skeleton:
#         color = ske[2]
#         cv2.line(frame, (int(key_points[fra][ske[0]][0]), int(key_points[fra][ske[0]][1])),
#                  (int(key_points[fra][ske[1]][0]), int(key_points[fra][ske[1]][1])),
#                  color, 3)
#
#     fra += 1
#
#     cv2.imshow("girl.mp4", frame)
#     cv2.waitKey(30)



