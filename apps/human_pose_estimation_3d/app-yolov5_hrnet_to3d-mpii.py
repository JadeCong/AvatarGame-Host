from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import time
from pathlib import Path
import csv
import shutil
import yaml
from easydict import EasyDict as edict
# from pynput import keyboard as kb
import keyboard as kb

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torchvision.transforms as transforms
import torchvision
from thop import profile

import cv2
from PIL import Image
import numpy as np
from numpy import random

sys.path.append("../../nns")
sys.path.append("../../nns/yolov5")
import yolov5
import yolov5.models as detect_models
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    plot_one_box, strip_optimizer, set_logging, increment_dir
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized

sys.path.append("../../nns/hrnet")
import hrnet
import hrnet.models as pose_models
from hrnet.config import cfg
from hrnet.config import update_config
from hrnet.core.inference_new import get_final_preds
from hrnet.utils.transforms import get_affine_transform

sys.path.append("..")
# from To3d.run import Gen3Dpose

sys.path.append("../../utils")
from filter.one_euro_filter import OneEuroFilter
from goto_python3.goto import goto


# define the configurations for MPII keypoints and skeletons
MPII_KEYPOINT_INDEXES = {
    0: 'right_ankle',
    1: 'right_knee',
    2: 'right_hip',
    3: 'left_hip',
    4: 'left_knee',
    5: 'left_ankle',
    6: 'pelvis',
    7: 'thorax',
    8: 'upper_neck',
    9: 'head_top',
    10: 'right_wrist',
    11: 'right_elbow',
    12: 'right_shoulder',
    13: 'left_shoulder',
    14: 'left_elbow',
    15: 'left_wrist'
}

MPII_SKELETON_COLORS = {
    "color1": [0, 0, 200],
    "color2": [0, 255, 0],
    "color3": [255, 0, 0],
    "color4": [10, 100, 255],
    "color5": [100, 0, 200]
}

MPII_SKELETON_INDEXES = [
    [0, 1, MPII_SKELETON_COLORS["color1"]], [1, 2, MPII_SKELETON_COLORS["color1"]], [2, 7, MPII_SKELETON_COLORS["color1"]], 
    [5, 4, MPII_SKELETON_COLORS["color2"]], [4, 3, MPII_SKELETON_COLORS["color2"]], [3, 7, MPII_SKELETON_COLORS["color2"]], 
    [10, 11, MPII_SKELETON_COLORS["color3"]], [11, 12, MPII_SKELETON_COLORS["color3"]], [12, 7, MPII_SKELETON_COLORS["color3"]], 
    [15, 14, MPII_SKELETON_COLORS["color4"]], [14, 13, MPII_SKELETON_COLORS["color4"]], [13, 7, MPII_SKELETON_COLORS["color4"]], 
    [7, 8, MPII_SKELETON_COLORS["color5"]], [8, 9, MPII_SKELETON_COLORS["color5"]]
]

PACIFIC_SKELETON_INDEXES = [
    [0, 1, MPII_SKELETON_COLORS["color1"]], [1, 2, MPII_SKELETON_COLORS["color1"]], [2, 6, MPII_SKELETON_COLORS["color1"]], 
    [5, 4, MPII_SKELETON_COLORS["color2"]], [4, 3, MPII_SKELETON_COLORS["color2"]], [3, 6, MPII_SKELETON_COLORS["color2"]], 
    [10, 11, MPII_SKELETON_COLORS["color3"]], [11, 12, MPII_SKELETON_COLORS["color3"]], [12, 7, MPII_SKELETON_COLORS["color3"]], 
    [15, 14, MPII_SKELETON_COLORS["color4"]], [14, 13, MPII_SKELETON_COLORS["color4"]], [13, 7, MPII_SKELETON_COLORS["color4"]], 
    [6, 7, MPII_SKELETON_COLORS["color5"]], [7, 8, MPII_SKELETON_COLORS["color5"]], [8, 9, MPII_SKELETON_COLORS["color5"]]
]


def parseArgs():
    parser = argparse.ArgumentParser(description='Process the argument inputs...')
    
    parser.add_argument('--config', type=str, required=True, help="Path, where config file is stored")
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--local_rank', type=int, help="Local rank of the process on the node")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5l.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--view_img', action='store_true', help='display results')
    parser.add_argument('--save_txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save_conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save_dir', type=str, default='runs/detect', help='directory to save results')
    parser.add_argument('--name', default='', help='name to append to --save-dir: i.e. runs/{N} -> runs/{N}_{name}')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    
    args = parser.parse_args()
    
    return args


def initDistributed(args):
    """Initialize the configuration for distributed network training
    
    Args:
        args ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    # check the number of aviliable gpu for whether using distributed training
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:
        print("The hardware capacity of device cannot support distributed training.")
        return False
    
    # Set the local_rank for master node
    torch.cuda.set_device(args.local_rank)
    assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
    assert os.environ["RANK"], "use pytorch launcher and explicityly state the rank of the process"
    print("The hardware capacity of device can support distributed calculation.")
    
    # Set the seeds for distributed GPUs
    torch.manual_seed(args.seed)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    print("The distributed calculation has been initialized.")
    
    return True


def loadConfig(cfg_path):
    """Load the configuration for network and model
    
    Args:
        cfg_path ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    with open(cfg_path) as fin:
        config = edict(yaml.safe_load(fin))
    
    return config


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int
    
    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)
    
    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5
    
    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200
    
    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25
    
    return center, scale


def get_pose_estimation_prediction(config, pose_model, image, centers, scales, transform, device):
    """Predict the 2D human pose using the hrnet model
    
    Args:
        config ([type]): [description]
        pose_model ([type]): [description]
        image ([type]): [description]
        centers ([type]): [description]
        scales ([type]): [description]
        transform ([type]): [description]
        device ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    rotation = 0
    
    # pose estimation transformation
    model_inputs = []
    for center, scale in zip(centers, scales):
        trans = get_affine_transform(center, scale, rotation, config.MODEL.HRNET.IMAGE_SIZE)
        # Crop smaller image of people
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(config.MODEL.HRNET.IMAGE_SIZE[0]), int(config.MODEL.HRNET.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)
        
        # hwc -> 1chw
        model_input = transform(model_input)#.unsqueeze(0)
        model_inputs.append(model_input)
    
    # n * 1chw -> nchw
    model_inputs = torch.stack(model_inputs)
    
    # compute output heatmap
    output = pose_model(model_inputs.to(device)) # get the heatmap of human box
    coords, _ = get_final_preds(
        config,
        output.cpu().detach().numpy(),
        np.asarray(centers),
        np.asarray(scales))
    
    return coords


def constructFilters(data_shape, freq=25, mincutoff=1, beta=1, dcutoff=1):
    """Construct the filters for 2D or 3D estimation
    
    Args:
        data_shape ([type]): [description]
        freq (int, optional): [description]. Defaults to 25.
        mincutoff (int, optional): [description]. Defaults to 1.
        beta (int, optional): [description]. Defaults to 1.
        dcutoff (int, optional): [description]. Defaults to 1.
    
    Raises:
        Exception: [description]
    
    Returns:
        [type]: [description]
    """
    if len(data_shape) == 3:
        # define the filter array
        filters = [[None] * data_shape[1] for _ in range(data_shape[0])]
        # construct coordinate meta filters
        if data_shape[2] == 2:
            filter = [OneEuroFilter(freq=freq, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff), 
                      OneEuroFilter(freq=freq, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)]
        elif data_shape[2] == 3:
            filter = [OneEuroFilter(freq=freq, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff), 
                      OneEuroFilter(freq=freq, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff),
                      OneEuroFilter(freq=freq, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)]
        # construct filter array
        for person in range(data_shape[0]):
            for kp in range(data_shape[1]):
                filters[person][kp] = filter
        
        return filters
    else:
        raise Exception("Wrong Data Shape for Constructing Filters.", data_shape)


def on_press(key):
    """Callback function for listener of keyboard event
    
    Args:
        key ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    if key == None:
        print('WTF')
        pass
    elif key != kb.Key.esc:
        print('=> %s' % key)
        print("No Human Detected and Press Esc Key to Quit.")
        return False
    elif key == kb.Key.esc:
        print('=> %s' % key)
        print("No Human Detected and Quit with Esc Key.")
        return False


def mainFunc(args):
    # Set the main function flag
    print("Main Function Start...")
    
    # Check the GPU device
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))
    
    # Check whether using the distributed runing for the network
    is_distributed = initDistributed(args)
    master = True
    if is_distributed and os.environ["RANK"]:
        master = int(os.environ["RANK"]) == 0  # check whether this node is master node
    
    # Configuration for device setting
    set_logging()
    if is_distributed:
        device = torch.device('cuda:{}'.format(args.local_rank))
    else:
        device = select_device(args.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load the configuration
    config = loadConfig(args.config)
    
    # CuDNN related setting
    if torch.cuda.is_available():
        cudnn.benchmark = config.DEVICE.CUDNN.BENCHMARK
        cudnn.deterministic = config.DEVICE.CUDNN.DETERMINISTIC
        cudnn.enabled = config.DEVICE.CUDNN.ENABLED
    
    # Configurations for dirctories
    save_img, save_dir, source, yolov5_weights, view_img, save_txt, imgsz = \
        False, Path(args.save_dir), args.source, args.weights, args.view_img, args.save_txt, args.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')
    
    if save_dir == Path('runs/detect'):  # if default
        os.makedirs('runs/detect', exist_ok=True)  # make base
        save_dir = Path(increment_dir(save_dir / 'exp', args.name))  # increment run
    os.makedirs(save_dir / 'labels' if save_txt else save_dir, exist_ok=True)  # make new dir
    
    # Load yolov5 model for human detection
    model_yolov5 = attempt_load(config.MODEL.PRETRAINED.YOLOV5, map_location=device)
    imgsz = check_img_size(imgsz, s=model_yolov5.stride.max())  # check img_size
    if half:
        model_yolov5.half()  # to FP16
    
    # Second-stage classifier
    classify = False
    if classify:
        model_classifier = load_classifier(name='resnet101', n=2)  # initialize
        model_classifier.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        model_classifier.to(device).eval()
    
    # Load hrnet model for human keypoints estimation
    model_hrnet = eval('pose_models.'+config.MODEL.NAME.HRNET+'.get_pose_net')(config, is_train=False)
    if config.EVAL.HRNET.MODEL_FILE:
        print('=> loading model from {}'.format(config.EVAL.HRNET.MODEL_FILE))
        model_hrnet.load_state_dict(torch.load(config.EVAL.HRNET.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at EVAL.HRNET.MODEL_FILE')
    model_hrnet.to(device)
    model_hrnet.eval()
    
    # Create the 3d human pose mapping processor
    # pose_mapper = Gen3Dpose()
    
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
    pose_transform = transforms.Compose([  # input transformation for 2d human pose estimation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Get names and colors
    names = model_yolov5.module.names if hasattr(model_yolov5, 'module') else model_yolov5.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    
    # Construt filters for filtering 2D/3D human keypoints
    # filters_2d = constructFilters((1,16,2), freq=25, mincutoff=1, beta=0.01)  # for test
    # filters_3d = constructFilters((1,16,3), freq=25, mincutoff=1, beta=0.01)
    
    # Run the yolov5 and hrnet for 2d human pose estimation
    # with torch.no_grad():
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model_yolov5(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    # Process every video frame
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        t1 = time_synchronized()
        pred_boxes = model_yolov5(img, augment=args.augment)[0]
        
        # Apply NMS
        pred_boxes = non_max_suppression(pred_boxes, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=args.agnostic_nms)
        t2 = time_synchronized()
        
        # Can not find people and move to next frame
        if pred_boxes[0] is None:
            # show the frame with no human detected
            cv2.namedWindow("3D Human Pose Estimation", cv2.WINDOW_NORMAL)
            cv2.imshow("3D Human Pose Estimation", im0s[0].copy())
            # wait manual operations
            # with kb.Listener(on_press=on_press) as listener:
            #     listener.join()
            #     return
            # if kb.is_pressed('t'):
            #     return
            print("No Human Detected and Move on.")
            print("-" * 30)
            continue
        
        # Print time (inference + NMS)
        detect_time = t2 - t1
        detect_fps = 1.0 / detect_time
        print("Human Detection Time: {}, Human Detection FPS: {}".format(detect_time, detect_fps))
        
        # Apply Classifier
        if classify:  # false
            pred_boxes = apply_classifier(pred_boxes, model_classifier, img, im0s)
        
        # Estimate 2d human pose(multiple person)
        centers = []
        scales = []
        for id, boxes in enumerate(pred_boxes):
            if boxes is not None and len(boxes):
                boxes[:, :4] = scale_coords(img.shape[2:], boxes[:, :4], im0s[id].copy().shape).round()
            # convert tensor to list format
            boxes = np.delete(boxes.cpu().numpy(), [-2,-1], axis=1).tolist()
            for l in range(len(boxes)):
                boxes[l] = [tuple(boxes[l][0:2]), tuple(boxes[l][2:4])]
            # convert box to center and scale
            for box in boxes:
                center, scale = box_to_center_scale(box, imgsz, imgsz)
                centers.append(center)
                scales.append(scale)
        t3 = time_synchronized()
        pred_pose_2d = get_pose_estimation_prediction(config, model_hrnet, im0s[0], centers, scales, transform=pose_transform, device=device)
        t4 = time_synchronized()
        
        # Print time (2d human pose estimation)
        estimate_time = t4 - t3
        estimate_fps = 1.0 / estimate_time
        print("Pose Estimation Time: {}, Pose Estimation FPS: {}".format(estimate_time, estimate_fps))
        
        # Filter the predicted 2d human pose(multiple person)
        t5 = time_synchronized()
        # if False:  # for test
        if config.EVAL.HRNET.USE_FILTERS_2D:
            # construct filters for every keypoints of every person in 2D
            filters_2d = constructFilters(pred_pose_2d.shape, freq=8, mincutoff=8, beta=0.001)
            print("Shape of filters_2d: ({}, {}, {})".format(len(filters_2d), len(filters_2d[0]), len(filters_2d[0][0])))  # for test
            for per in range(pred_pose_2d.shape[0]):
                for kp in range(pred_pose_2d.shape[1]):
                    for coord in range(pred_pose_2d.shape[2]):
                        pred_pose_2d[per][kp][coord] = filters_2d[per][kp][coord](pred_pose_2d[per][kp][coord])
        t6 = time_synchronized()
        
        # Print time (filter 2d human pose)
        filter_time_2d = t6 - t5
        filter_fps_2d = 1.0 / filter_time_2d
        print("Filter 2D Pose Time: {}, Filter 2D Pose FPS: {}".format(filter_time_2d, filter_fps_2d))
        
        # Process detections and estimations in 2D
        for i, box in enumerate(pred_boxes):
            if webcam:  # batch_size >= 1
                p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = Path(path), '', im0s
            
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if box is not None and len(box):
                # Rescale boxes from img_size to im0 size
                box[:, :4] = scale_coords(img.shape[2:], box[:, :4], im0.shape).round()
                
                # Print results
                for c in box[:, -1].unique():
                    n = (box[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(box):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if args.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line) + '\n') % line)
                    
                    # Add bbox to image
                    if save_img or view_img:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                
                # Draw joint keypoints, number orders and human skeletons for every detected people in 2D
                for person in pred_pose_2d:
                    # draw the human keypoints
                    for idx, coord in enumerate(person):
                        x_coord, y_coord = int(coord[0]), int(coord[1])
                        cv2.circle(im0, (x_coord, y_coord), 1, (0, 0, 255), 5)
                        cv2.putText(im0, str(idx), (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # draw the human skeletons in PACIFIC mode
                    for skeleton in PACIFIC_SKELETON_INDEXES:
                        cv2.line(im0, (int(person[skeleton[0]][0]), int(person[skeleton[0]][1])), (int(person[skeleton[1]][0]), int(person[skeleton[1]][1])), skeleton[2], 2)
            
            # Print time (inference + NMS + estimation)
            print('%sDone. (%.3fs)' % (s, t4 - t1))
            
            # Stream results
            if view_img:
                detect_text = "Detect FPS:{0:0>5.2f}/{1:0>6.2f}ms".format(detect_fps, detect_time*1000)
                estimate_text = "Estimate FPS:{0:0>5.2f}/{1:0>6.2f}ms".format(estimate_fps, estimate_time*1000)
                cv2.putText(im0, detect_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(im0, estimate_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.namedWindow("3D Human Pose Estimation", cv2.WINDOW_NORMAL)
                cv2.imshow("3D Human Pose Estimation", im0)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # q to quit
                    return
                    # goto .mainFunc
            
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        
                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
        
        # Mapping 3d human pose
        t7 = time_synchronized()
        mapping_pose_3d = None
        t8 = time_synchronized()
        
        # Print time (3d human pose mapping)
        mapping_time = t8 - t7
        mapping_fps = 1.0 / mapping_time
        print("Pose Mapping Time: {}, Pose Mapping FPS: {}".format(mapping_time, mapping_fps))
        
        # Filter the mapping 3d human pose
        t9 = time_synchronized()
        # if False:  # for test
        # if config.EVAL.TO3D.USE_FILTERS_3D:
        #     # construct filters for every keypoints of every person in 3D
        #     filters_3d = constructFilters(mapping_pose_3d.shape, freq=1, mincutoff=1, beta=0.01)
        #     print("Shape of filters_3d: ({}, {}, {})".format(len(filters_3d), len(filters_3d[0]), len(filters_3d[0][0])))  # for test
        #     for per in range(mapping_pose_3d.shape[0]):
        #         for kp in range(mapping_pose_3d.shape[1]):
        #             for coord in range(mapping_pose_3d.shape[2]):
        #                 mapping_pose_3d[per][kp][coord] = filters_3d[per][kp][coord](mapping_pose_3d[per][kp][coord])
        t10 = time_synchronized()
        
        # Print time (filter 3d human pose)
        filter_time_3d = t10 - t9
        filter_fps_3d = 1.0 / filter_time_3d
        print("Filter 3D Pose Time: {}, Filter 3D Pose FPS: {}".format(filter_time_3d, filter_fps_3d))
        
        # Print time (inference + NMS + estimation + mapping + 2d/3d filtering)
        all_process_time = t10 - t1
        all_process_fps = 1.0 / all_process_time
        print("All Process Time: {}, All Process FPS: {}".format(all_process_time, all_process_fps))
        print("-" * 30)
    
    # Goto label
    # label .mainFunc
    
    # Print saving results
    if save_txt or save_img:
        print('Results saved to %s' % save_dir)
    
    # Release video reader and writer, then destory all opencv windows
    dataset.vid_cap.release()
    vid_writer.release()
    cv2.destroyAllWindows()
    print('Present 3D Human Pose Inference Done. Total Time:(%.3f seconds)' % (time.time() - t0))


if __name__ == '__main__':
    args = parseArgs()
    print("Argument Inputs: {}".format(args))
    mainFunc(args)
    print("All Done!")
