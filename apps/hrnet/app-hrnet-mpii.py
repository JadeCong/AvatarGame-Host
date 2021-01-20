from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np

import sys
sys.path.append("../../nns/hrnet")
import time

# import _init_paths
import models
from config import cfg
from config import update_config
from core.inference import get_final_preds
from utils.transforms import get_affine_transform

# calculate the amount of network parameters and operations
from thop import profile


# define the GPU aviable flag
CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# pytorch official category for object detection
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

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


def get_person_detection_boxes(model, img, threshold=0.5):
    pil_image = Image.fromarray(img)  # Load the image
    transform = transforms.Compose([transforms.ToTensor()])  # Defing PyTorch Transform
    transformed_img = transform(pil_image)  # Apply the transform to the image
    pred = model([transformed_img.to(CTX)])  # Pass the image to the model
    # Use the first detected person
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].cpu().detach().numpy())]  # Bounding boxes
    pred_scores = list(pred[0]['scores'].cpu().detach().numpy())

    person_boxes = []
    # Select box has score larger than threshold and is person
    for pred_class, pred_box, pred_score in zip(pred_classes, pred_boxes, pred_scores):
        if (pred_score > threshold) and (pred_class == 'person'):
            person_boxes.append(pred_box)

    return person_boxes


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


def get_pose_estimation_prediction(pose_model, image, centers, scales, transform):
    rotation = 0

    # pose estimation transformation
    model_inputs = []
    for center, scale in zip(centers, scales):
        trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
        # Crop smaller image of people
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        # hwc -> 1chw
        model_input = transform(model_input)#.unsqueeze(0)
        model_inputs.append(model_input)

    # n * 1chw -> nchw
    model_inputs = torch.stack(model_inputs)

    # compute output heatmap
    output = pose_model(model_inputs.to(CTX)) # get the heatmap of human box
    coords, _ = get_final_preds(
        cfg,
        output.cpu().detach().numpy(),
        np.asarray(centers),
        np.asarray(scales))

    return coords


def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    return pose_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    # parser.add_argument('--videoFile', type=str, required=True)
    parser.add_argument('--videoFile', type=int, required=True, default=0)
    parser.add_argument('--outputDir', type=str, default='/output/')
    # parser.add_argument('--inferenceFps', type=int, default=10)
    parser.add_argument('--inferenceFps', type=int, default=30)
    parser.add_argument('--writeBoxFrames', action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)
    pose_dir = prepare_output_dirs(args.outputDir)
    csv_output_rows = []

    # load the object detection network 
    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(CTX)
    box_model.eval()

    # load the human pose detection network
    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )
    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')
    pose_model.to(CTX)
    pose_model.eval()

    # Loading an video
    vidcap = cv2.VideoCapture(args.videoFile)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps < args.inferenceFps:
        print('desired inference fps is '+str(args.inferenceFps)+' but video fps is '+str(fps))
        exit()
    skip_frame_cnt = round(fps / args.inferenceFps)
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # outcap = cv2.VideoWriter('{}/{}_pose.avi'.format(args.outputDir, os.path.splitext(os.path.basename(args.videoFile))[0]),
    #                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int(skip_frame_cnt), (frame_width, frame_height))
    # outcap = cv2.VideoWriter('{}/{}_pose.avi'.format(args.outputDir, 'human'),
    #                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int(skip_frame_cnt), (frame_width, frame_height))

    # calculate the network parameter amount and operation amount
    # TODO: test the thop function
    # _, image_bgr = vidcap.read()
    # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # image_per = image_rgb.copy()
    # image_per = Image.fromarray(image_per)    
    # image_per = transforms.Compose([transforms.ToTensor()])(image_per)
    # box_macs, box_params = profile(box_model, inputs=(image_per, ))
    # print("The human box parameter amount: {}".format(box_params))
    # print("The human box operation amount: {}".format(box_macs))

    # predict the human box and human pose
    count = 0
    while vidcap.isOpened():
        total_now = time.time()
        ret, image_bgr = vidcap.read()
        count += 1

        if not ret:
            continue

        if count % skip_frame_cnt != 0:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Clone 2 image for person detection and pose estimation
        if cfg.DATASET.COLOR_RGB:
            image_per = image_rgb.copy()
            image_pose = image_rgb.copy()
        else:
            image_per = image_bgr.copy()
            image_pose = image_bgr.copy()

        # Clone 1 image for debugging purpose
        image_debug = image_bgr.copy()

        # object detection box
        now = time.time()
        pred_boxes = get_person_detection_boxes(box_model, image_per, threshold=0.9)
        then = time.time()
        print("Find person bbox in: {} sec".format(then - now))

        # Can not find people. Move to next frame
        if not pred_boxes:
            count += 1
            continue
        
        # draw the bounding box
        if args.writeBoxFrames:
            for box in pred_boxes:
                cv2.rectangle(image_debug, box[0], box[1], color=(0, 255, 0), thickness=1)  # Draw Rectangle with the coordinates

        # pose estimation : for multiple people
        centers = []
        scales = []
        for box in pred_boxes:
            print("box: {}".format(box))
            center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            centers.append(center)
            scales.append(scale)

        now = time.time()
        pose_preds = get_pose_estimation_prediction(pose_model, image_pose, centers, scales, transform=pose_transform)
        then = time.time()
        print("Find person pose in: {} sec".format(then - now))

        # draw joint keypoints and skeletons for every detected people
        new_csv_row = []
        for coords in pose_preds:
            # Draw each point on image and connect them for skeletons
            for idx, coord in enumerate(coords):
                x_coord, y_coord = int(coord[0]), int(coord[1])
                cv2.circle(image_debug, (x_coord, y_coord), 1, (0, 0, 255), 5)
                cv2.putText(image_debug, str(idx), (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                new_csv_row.extend([x_coord, y_coord])
            
            # draw the human skeletons in MPII mode
            # for skeleton in MPII_SKELETON_INDEXES:
            #     cv2.line(image_debug, (int(coords[skeleton[0]][0]), int(coords[skeleton[0]][1])), (int(coords[skeleton[1]][0]), int(coords[skeleton[1]][1])), skeleton[2], 2)
            
            # draw the human skeletons in PACIFIC mode
            for skeleton in PACIFIC_SKELETON_INDEXES:
                cv2.line(image_debug, (int(coords[skeleton[0]][0]), int(coords[skeleton[0]][1])), (int(coords[skeleton[1]][0]), int(coords[skeleton[1]][1])), skeleton[2], 2)
            

        # calculate the estimation time and FPS
        total_then = time.time()
        detect_time = total_then - total_now
        detect_fps = 1.0 / detect_time
        print("2D Human Pose Estimation in FPS: {:.2f}".format(detect_fps))

        # put the needed infos(FPS/time/text) on the image stream
        text = "FPS:{0:0>5.2f}/{1:0>6.2f}ms".format(detect_fps, detect_time*1000)
        cv2.putText(image_debug, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # show the image stream in window
        cv2.namedWindow("2D Human Pose Estimation", cv2.WINDOW_NORMAL)
        cv2.imshow("2D Human Pose Estimation", image_debug)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # save the images and video
        csv_output_rows.append(new_csv_row)
        img_file = os.path.join(pose_dir, 'pose_{:08d}.jpg'.format(count))
        cv2.imwrite(img_file, image_debug)
        # outcap.write(image_debug)

    # write csv
    csv_headers = ['frame']
    for keypoint in MPII_KEYPOINT_INDEXES.values():
        csv_headers.extend([keypoint+'_x', keypoint+'_y'])

    csv_output_filename = os.path.join(args.outputDir, 'pose-data.csv')
    with open(csv_output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_headers)
        csvwriter.writerows(csv_output_rows)

    vidcap.release()
    # outcap.release()

    cv2.destroyAllWindows()
    print("Live Video Done.")


if __name__ == '__main__':
    main()
