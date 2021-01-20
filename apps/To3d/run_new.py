# from To3d.utils.model import TemporalModel
# from To3d.utils.generators import UnchunkedGenerator
# from To3d.utils.utils import wrap
# from To3d.utils.gen_keypoint import gen_kepoint_file
import os, sys
sys.path.append("..")
from To3d.utils.model import TemporalModel
from To3d.utils.generators import UnchunkedGenerator
from To3d.utils.utils import wrap
from To3d.utils.gen_keypoint import gen_kepoint_file
import torch
import time
import numpy as np
import argparse
import json


def argparse_args():
    parser = argparse.ArgumentParser(description='Training script')
    
    # General arguments
    parser.add_argument('--c', '--checkpoint', default='./To3d/checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('--r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('--evaluate', default='epoch_120.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--render', default=True, action='store_true', help='visualize a particular video')
    parser.add_argument('--export-training-curves', action='store_true', help='save training curves as .png images')
    
    # Model arguments
    parser.add_argument('--drop', '--dropout', default=0.25, type=float, metavar='P', help='dropout probability')
    parser.add_argument('--arc', '--architecture', default='3,3,3,3,3', type=str, metavar='LAYERS', help='filter widths separated by comma')
    parser.add_argument('--causal', default=True, action='store_true', help='use causal convolutions for real-time processing')
    parser.add_argument('--ch', '--channels', default=1024, type=int, metavar='N', help='number of channels in convolution layers')
    # Experimental
    parser.add_argument('--dense', action='store_true', help='use dense convolutions instead of dilated convolutions')
    parser.add_argument('--txt_path', type=str, default='./To3d/data/camera_test.npy', metavar='PATH', help='video per frams keypoints info txt')
    # Visualization
    parser.add_argument('--viz_video', type=str, default='./facebook3d/video/camera_test.avi', metavar='PATH', help='path to input video')
    parser.add_argument('--viz_skip', type=int, default=0, metavar='N', help='skip first N frames of input video')
    parser.add_argument('--viz_output', type=str, default='output.mp4', metavar='PATH', help='output file name (.gif or .mp4)')
    parser.add_argument('--viz_bitrate', type=int, default=3000, metavar='N', help='bitrate for mp4 videos')
    parser.add_argument('--viz_limit', type=int, default=1000, metavar='N', help='only render first N frames')
    parser.add_argument('--viz_downsample', type=int, default=0, metavar='N', help='downsample FPS by a factor N')
    parser.add_argument('--viz_size', type=int, default=3, metavar='N', help='image size')
    parser.add_argument('--viz_len', type=int, default=1000, metavar='N', help='video frams nums')
    parser.add_argument('--viz_width', type=int, default=640, metavar='N', help='video width')
    parser.add_argument('--viz_height', type=int, default=480, metavar='N', help='video height')
    
    parser.set_defaults(bone_length_term=True)
    parser.set_defaults(data_augmentation=True)
    parser.set_defaults(test_time_augmentation=True)
    
    args = parser.parse_args()
    
    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()
    
    if args.export_training_curves and args.no_eval:
        print('Invalid flags: --export-training-curves and --no-eval cannot be set at the same time')
        exit()
    
    return args

class Arguments():
    def __init__(self, args_dict):
        self.dict = args_dict
        self.checkpoint = self.dict["checkpoint"]
        self.resume = self.dict["resume"]
        self.evaluate = self.dict["evaluate"]
        self.render = self.dict["render"]
        self.export_training_curves = self.dict["export_training_curves"]
        self.dropout = self.dict["dropout"]
        self.architecture = self.dict["architecture"]
        self.causal = self.dict["causal"]
        self.channels = self.dict["channels"]
        self.dense = self.dict["dense"]
        self.txt_path = self.dict["txt_path"]
        self.viz_video = self.dict["viz_video"]
        self.viz_skip = self.dict["viz_skip"]
        self.viz_output = self.dict["viz_output"]
        self.viz_bitrate = self.dict["viz_bitrate"]
        self.viz_limit = self.dict["viz_limit"]
        self.viz_downsample = self.dict["viz_downsample"]
        self.viz_size = self.dict["viz_size"]
        self.viz_len = self.dict["viz_len"]
        self.viz_width = self.dict["viz_width"]
        self.viz_height = self.dict["viz_height"]
        self.bone_length_term = self.dict["bone_length_term"]
        self.data_augmentation = self.dict["data_augmentation"]
        self.test_time_augmentation = self.dict["test_time_augmentation"]


# export the to3d_args
# args = argparse_args()
# with open('to3d_args.txt', 'w') as fout:
#     json.dump(args.__dict__, fout, indent=2)

# import the to3d_args
to3d_args_dict = dict()
with open('../To3d/to3d_args.txt', 'r') as fin:
    to3d_args_dict = json.load(fin)
print(to3d_args_dict)
to3d_args = Arguments(to3d_args_dict)


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape)-1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape)-1)
    return (v + 2 * (q[..., :1] * uv + uuv))

def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t


class Gen3Dpose(object):
    def __init__(self, args):
        self.args = args
        self.data_path = '../To3d/data/camera_test.npy'
        self.video_len = self.args.viz_len
        self.input = self.load_data()
        self.filter_widths = [int(x) for x in args.architecture.split(',')]
        self.init_net_prams()
    
    def load_data(self):
        skelen = np.load(self.data_path)
        # skelen = gen_kepoint_file(self.data_path,self.video_len)
        skelen[:, :, 0] *= 1000 / self.args.viz_width
        skelen[:, :, 1] *= 1000 / self.args.viz_height
        # Normalize camera frame
        skelen[..., :2] = normalize_screen_coordinates(skelen[..., :2], w=1000, h=1000)
        
        return skelen
    
    def normalize_data(self,skelen):
        skelen[:, 0] *= 1000 / self.args.viz_width
        skelen[:, 1] *= 1000 / self.args.viz_height
        # Normalize camera frame
        skelen[..., :2] = normalize_screen_coordinates(skelen[..., :2], w=1000, h=1000)
        
        return skelen
    
    def init_net_prams(self):
        model_pos = TemporalModel(self.input.shape[-2], self.input.shape[-1], self.input.shape[-2],
                                       filter_widths=self.filter_widths, causal=self.args.causal, dropout=self.args.dropout,
                                       channels=self.args.channels, dense=self.args.dense)
        
        if torch.cuda.is_available():
            self.model_pos = model_pos.cuda()
        
        receptive_field = self.model_pos.receptive_field()
        print('INFO: Receptive field: {} frames'.format(receptive_field))
        pad = (receptive_field - 1) // 2  # Padding on each side
        if self.args.causal:
            print('INFO: Using causal convolutions')
            causal_shift = pad
        else:
            causal_shift = 0
        self.pad = pad
        self.causal_shift = causal_shift
        
        if self.args.resume or self.args.evaluate:
            chk_filename = os.path.join(self.args.checkpoint, self.args.resume if self.args.resume else self.args.evaluate)
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            print('This model was trained for {} epochs'.format(checkpoint['epoch']))
            self.model_pos.load_state_dict(checkpoint['model_pos'])
    
    def gen_data(self):
        kps_left = [4, 5, 6, 11, 12, 13]
        kps_right = [1, 2, 3, 14, 15, 16]
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        gen = UnchunkedGenerator(None, None, [self.input],
                                 pad=self.pad, causal_shift=self.causal_shift,
                                 augment=self.args.test_time_augmentation,
                                 kps_left=kps_left, kps_right=kps_right,
                                 joints_left=joints_left,
                                 joints_right=joints_right)
        return gen
    
    def cur_2d_pose(self,cur_2dpose, pass_2dinputs):
        kps_left = [4, 5, 6, 11, 12, 13]
        kps_right = [1, 2, 3, 14, 15, 16]
        
        cur_inputs = np.zeros([1, 244, 17, 2])
        cur_inputs[:, :243, :, :] = pass_2dinputs
        cur_inputs[:, 243, :, :] = cur_2dpose
        
        cur_inputs = np.delete(cur_inputs, 0, axis=1)
        cur_inputs = np.concatenate((cur_inputs, cur_inputs), axis=0)
        cur_inputs[1, :, :, 0] *= -1
        cur_inputs[1, :, kps_left + kps_right] = cur_inputs[1, :, kps_right + kps_left]
        
        return cur_inputs
    
    def get_prediction(self):
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        rot = np.array([0.15232472, -0.1544232, -0.75475633, 0.619107], np.float32)
        
        with torch.no_grad():
            
            self.model_pos.eval()
            
            for _, batch, batch_2d in self.gen_data().next_epoch():
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                
                # Positional model
                predicted_3d_pos, predicted_traj = self.model_pos(inputs_2d)
                # predicted_traj[:, :, 0, 2] = 0
                # for i in range(17):
                #     predicted_3d_pos[:, :, i, :] += predicted_traj[:, :, 0, :]
                
                # Test-time augmentation (if enabled)
                if self.gen_data().augment_enabled():
                    # Undo flipping and take average with non-flipped version
                    predicted_3d_pos[1, :, :, 0] *= -1
                    predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :,
                                                                         joints_right + joints_left]
                    predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
                
                prediction = predicted_3d_pos.squeeze(0).cpu().numpy()
                prediction = camera_to_world(prediction, R=rot, t=0)
                # We don't have the trajectory, but at least we can rebase the height
                # prediction[:, :, 2] -= np.min(prediction[:, :, 2])
                #
                # for i in range(len(prediction)):
                #     min = np.min(prediction[i, :, 2])
                #     prediction[i, :, 2] -= min
        
        return prediction
    
    def get_cur_prediction(self, batch_2d):
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        rot = np.array([0.15232472, -0.1544232, -0.75475633, 0.619107], np.float32)
        
        with torch.no_grad():
            
            self.model_pos.eval()
            
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
            
            # Positional model
            predicted_3d_pos, predicted_traj = self.model_pos(inputs_2d)
            predicted_traj[:, :, 0, 2] = 0
            for i in range(17):
                predicted_3d_pos[:, :, i, :] += predicted_traj[:, :, 0, :]
            
            # Test-time augmentation (if enabled)
            if self.gen_data().augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :,
                                                                     joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
            
            prediction = predicted_3d_pos.squeeze(0).cpu().numpy()
            prediction = camera_to_world(prediction, R=rot, t=0)
            # We don't have the trajectory, but at least we can rebase the height
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])
            
            # for i in range(len(prediction)):
            #     min = np.min(prediction[i, :, 2])
            #     prediction[i, :, 2] -= min
        
        return prediction
    
    def save_3dpose(self, file):
        with open('./gen3dpose.txt', 'w') as f:
            for i in range(file.shape[0]):
                for j in range(17):
                    f.write(str(j) + ' ' + str(file[i][j][0]) + ','
                            + str(file[i][j][1]) + ',' + str(file[i][j][2]) + ' ')
                f.write('\n')
            f.close()


# if args.render:
#     print('Rendering...')
#
#     gen_3dpose = Gen3Dpose()
#
#     per_2dpose = gen_3dpose.input
#
#     cur_2dpose = np.reshape(per_2dpose[0], [1, 17, 2])
#     inputs = cur_2dpose.repeat(243, axis=0)
#
#     fps = 0
#
#     per_pre3d = []
#     while True:
#         # start_t = time.time()
#
#         batch_2d = gen_3dpose.cur_2d_pose(per_2dpose[fps],inputs)
#         prediction = gen_3dpose.get_cur_prediction(batch_2d)
#         per_pre3d.append(prediction)
#
#         # end_t = time.time()
#         # print(fps,'Calculate the time each frame runs',end_t - start_t)
#         inputs = batch_2d[0]
#
#         fps += 1
#         if fps == per_2dpose.shape[0]:
#             break
#
#
#     # prediction = gen_3dpose.get_prediction()
#     # gen_3dpose.save_3dpose(prediction)
#
#     ########visualization#########
#     per_pre3d = np.reshape(per_pre3d,[fps,17,3])
#
#     anim_output = {'Reconstruction': per_pre3d}
#     anim_output['Ground truth'] = per_pre3d
#
#     # from utils.visualization import render_animation
#     #
#     #
#     # render_animation(np.load('./data/camera_test.npy'), anim_output,
#     #                  20, args.viz_bitrate, args.viz_output,
#     #                  limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
#     #                  input_video_path=args.viz_video, viewport=(1000, 1000),
#     #                  input_video_skip=args.viz_skip)#np.load('./pulati.npy')
#
#     ########visualization#########
#
#     from mpl_toolkits.mplot3d import axes3d
#     import matplotlib.pyplot as plt
#
#     # 打开画图窗口1，在三维空间中绘图
#
#
#     # 给出点（0，0，0）和（100，200，300）
#     for i in range(per_pre3d.shape[0]):
#
#         fig = plt.figure(1)
#         ax = fig.gca(projection='3d')
#         for j in range(16):
#             x = [per_pre3d[i][j+1][0], per_pre3d[i][j][0]]
#             y = [per_pre3d[i][j+1][1], per_pre3d[i][j][1]]
#             z = [per_pre3d[i][j+1][2], per_pre3d[i][j][2]]
#
#             # 将数组中的前两个点进行连线
#             figure = ax.plot(x, y, z,c='r')
#         plt.show()
