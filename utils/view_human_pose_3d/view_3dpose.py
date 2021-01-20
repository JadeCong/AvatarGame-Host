from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import os
import time




def view(prediction):
    fig = plt.figure(1)
    view_init = [45,45]
    for k in range(2):

        ax = fig.add_subplot(2, 1, k + 1, projection='3d')
        skelen_dex = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10]
            , [8, 14], [14, 15], [15, 16], [8, 11], [11, 12], [12, 13]]

        for i in range(len(skelen_dex)):
            x = [prediction[skelen_dex[i][0]][0], prediction[skelen_dex[i][1]][0]]
            y = [prediction[skelen_dex[i][0]][1], prediction[skelen_dex[i][1]][1]]
            z = [prediction[skelen_dex[i][0]][2], prediction[skelen_dex[i][1]][2]]
            ax.plot(x, y, z, c='b')

        ax.set_xlim3d([-2, 2])
        ax.set_zlim3d([0, 2])
        ax.set_ylim3d([-2, 3])
        ax.set_title('angle of view :' + str(view_init[k]))

        if k ==0:
             ax.view_init(0, view_init[k])
        else :
             ax.view_init(45, view_init[k])

    plt.pause(0.0001)
    plt.clf()


def view_3dpose(txt_path):
    while True:
        path_exists = os.path.exists(txt_path)
        if path_exists == True:
            prediction = []
            with open(txt_path, 'r') as f:
                line = f.readlines()
                if len(line) == 17 and len(line[16].split(',')) == 3:
                    for i in range(len(line)):
                        prediction.append(
                            [float(line[i].split(',')[0]), float(line[i].split(',')[1]), float(line[i].split(',')[2])])
                    prediction = np.reshape(prediction, [17, 3])
                else:
                    continue
            view(prediction)


        else:
            continue# print('skip 1 frames')



if __name__ == '__main__':
    txt_path = './SaveCur3d.txt'
    view_3dpose(txt_path)













