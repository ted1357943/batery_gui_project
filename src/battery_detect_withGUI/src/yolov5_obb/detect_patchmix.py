#!/usr/bin/env python
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from collections import defaultdict
import math

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, non_max_suppression_obb, print_args, scale_coords, scale_polys, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.rboxs_utils import poly2rbox, rbox2poly
from utils.augmentations import letterbox

from camera.realsense import RS, Aruco, convert_depth_to_phys_coord_using_realsense
from tools.Calculator import PolygonCalculator
import time as time
import random

# from tools.RobotArm import RobotController
#patchmix
sys.path.append('/home/iris/ting/yolov5_obb/patchmix')
import torch.nn as nn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from patchmix.config.finetune.vit_base_finetune import vit_base_finetune
from patchmix.config.finetune.vit_small_finetune import vit_small_finetune
from patchmix.config.finetune.vit_tiny_finetune import vit_tiny_finetune
from patchmix.module.vits import ViT
from PIL import Image

def predict_image_directly(image, model, device):
    # Define transformation for evaluation
    resize_im = 224 > 32
    transform_list = []
    if resize_im:
        size = int((256 / 224) * 224)  # Adjust the resizing factor
        transform_list.append(transforms.Resize(size, interpolation=3))
        transform_list.append(transforms.CenterCrop(224))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    transform = transforms.Compose(transform_list)

    # Transform the image
    image = image[:, :, [2, 1, 0]]
    image = Image.fromarray(image)

    transformed_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move image to the device
    transformed_image = transformed_image.to(device)

    # Set model to evaluation mode
    model.eval()
    # Disable gradient calculation
    with torch.no_grad():
        output = model(transformed_image)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.item()
    return pred

def send_start_signal(conveyor):
    print("start the conveyor belt")
    conveyor = True
    return conveyor

def send_stop_signal(conveyor):
    print("stop the conveyor belt")
    conveyor = False
    return conveyor

def center_distance(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def five_time_coculate(cylinder_info):
    staic_center_position=[]
    final_static_class_count=[]
    five_time_static_class=[]
    transposed_five_time_static_class=[]
    max_len_index=0
    
    for i in range(1, 5):
        if len(cylinder_info[i])>len(cylinder_info[max_len_index]):
            max_len_index = i
    max_len=len(cylinder_info[max_len_index])
    staic_cylinder_info = cylinder_info[max_len_index]
    
    for i in range(len(staic_cylinder_info)):
        staic_center_position.append(staic_cylinder_info[i][:2])
    
    five_time_static_class.append([item[2] for item in staic_cylinder_info])
    temp_class=[None]*max_len
    
    for i in range(5):
        if len(cylinder_info[i]) == max_len and i != max_len_index:
            for j in range(max_len):
                ori_center_x, ori_center_y = cylinder_info[i][j][:2]
                min_distance_index = 0
                min_distance = center_distance(ori_center_x, ori_center_y, *cylinder_info[max_len_index][0][:2])
                for k in range(1, max_len):
                    distance = center_distance(ori_center_x, ori_center_y, *cylinder_info[max_len_index][k][:2])
                    if distance < min_distance:
                        min_distance = distance
                        min_distance_index = k
                temp_class[min_distance_index] = cylinder_info[i][j][2]
            five_time_static_class.append(temp_class.copy())
    
    for i in range(max_len):
        transposed_inner = [five_time_static_class[j][i] for j in range(len(five_time_static_class))]
        transposed_five_time_static_class.append(transposed_inner)
    
    for items in transposed_five_time_static_class:
        class_counts = defaultdict(int)
        for item in items:
            if item is not None:
                class_counts[item] += 1
        
        most_common_class = max(class_counts, key=class_counts.get, default=None)
        final_static_class_count.append((most_common_class, class_counts[most_common_class] if most_common_class else 0))
    
    return staic_center_position, final_static_class_count

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        two_stage = False, # use two stage model
        grasp = False, # use grasp
        ):
    save_img = not nosave 
    webcam = source == '0' or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    if two_stage:
        two_stage_model = ViT(patch_size=16, img_size=224,
                    embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, drop_path_rate=0.1)
        two_stage_model.head = nn.Linear(two_stage_model.head.in_features, 163)
        two_stage_model.cuda(0)
        two_stage_weights = '/home/iris/ting/patchmix/out/vit-small_imagenet1k_epochs_100_lr_0.0005_battery-163/best.pth'
        two_stage_model = torch.load(two_stage_weights, map_location=torch.device(0))
        print("=> loaded two stage model '{}'".format(two_stage_weights))
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    vid_path, vid_writer, vid_cap = [None], [None],None
    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    # new_origin = np.array([0.72900003, -0.10555021, 0.04335386])
    new_origin = np.array([-0.003793996525928378, -0.10555021, 0.7320000529289246])
    cam_resolution = (1920, 1080)
    rs_cam = RS(cam_resolution[0], cam_resolution[1], frame_rate=30)

    conveyor = False
    conveyor = send_start_signal(conveyor)
    print(conveyor)
    while True:
        empty_count = 0
        count = 0
        while(conveyor==True):
            print("conveyor is running")
            rs_cam.update_raw_frames()
            color_frame = rs_cam.get_rgb_frame()
            hsv_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
            hue, saturation, value = cv2.split(hsv_frame)
            zero_channel = np.zeros_like(hue)*100
            merge_image = cv2.merge([hue, zero_channel, value])
            imgs = [None]
            im0s = [None]
            sources = [source]
            im0s[0] = color_frame
            imgs[0] = merge_image
            path = sources
            img = [letterbox(x, new_shape=imgsz, auto=pt)[0] for x in imgs]  # pad
            img = np.stack(img, 0)  # to numpy
            img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to 4D
            img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            # Get detections
            img = torch.from_numpy(img).to(device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img, augment=augment, visualize=visualize)
            pred = non_max_suppression_obb(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True, max_det=max_det)
            for i, det in enumerate(pred):  # per image
                im0 = im0s[i].copy()
                pred_poly = rbox2poly(det[:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
                pred_poly = scale_polys(img.shape[2:], pred_poly, im0.shape)
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                det = torch.cat((pred_poly, det[:, -2:]), dim=1)
                for *poly, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{names[c]} {conf:.2f}'
                    annotator.poly_label(poly, label, color=colors(c, True))
                for battery in pred_poly:
                    for i in range(0, 8, 2):
                        x, _ = battery[i], battery[i + 1]
                        if x < 350:
                            conveyor = send_stop_signal(conveyor)
                            break  # 停止检查该电池的其它顶点
            annotator.result()
            cv2.namedWindow('conveyor', cv2.WINDOW_NORMAL)
            cv2.imshow('conveyor', im0)
            if cv2.waitKey(1)== ord('q') : # 1 millisecond
                raise StopIteration
        time.sleep(2)
        while True:
            t0 = time.time()
            rs_cam.update_raw_frames()
            color_frame = rs_cam.get_rgb_frame()
            hsv_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
            hue, saturation, value = cv2.split(hsv_frame)
            zero_channel = np.zeros_like(hue)*100
            merge_image = cv2.merge([hue, zero_channel, value])
            imgs = [None]
            im0s = [None]
            sources = [source]
            im0s[0] = color_frame
            imgs[0] = merge_image
            path = sources
            img = [letterbox(x, new_shape=imgsz, auto=pt)[0] for x in imgs]  # pad
            img = np.stack(img, 0)  # to numpy
            img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to 4D
            img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            # Get detections
            img = torch.from_numpy(img).to(device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            t1 = time_sync()
            pred = model(img, augment=augment, visualize=visualize)    
            # Apply NMS
            pred = non_max_suppression_obb(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True, max_det=max_det)
            battery_tensor = pred[0]
            print("battery_tensor length:", len(battery_tensor))
            print("battery_tensor:", battery_tensor)
            if battery_tensor.numel()==0:
                empty_count+=1
                if empty_count<=50:
                    continue
                else:
                    print("No battery detected")
                    conveyor = send_start_signal(conveyor)
                    time.sleep(2)
                    break
            t2 = time_sync()
            # Process predictions
            no_battery = False
            for i, det in enumerate(pred):  # per image
                pred_poly = rbox2poly(det[:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
                seen += 1
                p, s, im0 = path[i], '%g: ', im0s[i].copy()
                s += f'{i}: '
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) # im.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                im_crop = im0.copy()
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale polys from img_size to im0 size
                    # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    pred_poly = scale_polys(img.shape[2:], pred_poly, im0.shape)
                    pred_length = torch.tensor([])
                    for i, poly in enumerate(pred_poly):
                        polys = poly.reshape(4,2).cpu().numpy()
                        calculator = PolygonCalculator()  
                        length = calculator.calculate_length(polys)
                        pred_length = torch.cat((pred_length, torch.tensor([length])))
                        ratio, longp1, longp2 = calculator.calculate_ratio(polys)
                        centerx, centery = calculator.calculate_center(polys)
                        new_center_x, new_center_y, new_center_z = convert_depth_to_phys_coord_using_realsense(centerx, centery, rs_cam.get_depth_value(int(centerx), int(centery)), rs_cam.intrinsics)
                        print(f"origin_center:{new_origin[0],new_origin[1]}")
                        print(f"new_center: {new_center_x, new_center_y}")
                        new_center_x = (new_center_x - new_origin[0])*1000
                        new_center_y = (new_center_y - new_origin[1])*1000
                        new_center_z = new_center_z - new_origin[2]
                        # center_tfx, center_tfy = calculator.calibration(centerx, centery)
                        longp1x, longp1y = longp1
                        longp2x, longp2y = longp2
                        longp1_tfx, longp1_tfy = calculator.calibration(longp1x, longp1y)
                        longp2_tfx, longp2_tfy = calculator.calibration(longp2x, longp2y)
                        angle = calculator.calculate_angle(float(longp1_tfx), float(longp1_tfy), float(longp2_tfx), float(longp2_tfy))
                        angle_str = str(angle)
                        print(f"Poligon {i+1}:")
                        print(f" pred_poly: {pred_poly[i]}")
                        print(f" center: {centerx, centery}")
                        # print(f" center_tf: {center_tfx, center_tfy}")
                        print(f"robot_move: {new_center_x, new_center_y}")
                    
                    det = torch.cat((pred_poly, det[:, -2:]), dim=1) # (n, [poly conf cls])
                    condition1 = pred_length <= 250
                    det = det[condition1]
                    last_column = det[:, -1]
                    condition2 = last_column != 0
                    det = det[condition2]
                    print(f"det:{det}")
                    if det.numel() == 0:
                        count+=1
                        if count<=50:
                            continue
                        else:
                            print("No battery detected")
                            no_battery = True
                            break
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        print(int(c))
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    classnumber = torch.tensor([])
                    for *poly, conf, cls in reversed(det):
                        c = int(cls)
                        if two_stage:
                            if(c==0):
                                label = f'{names[c]} {conf:.2f}'
                                annotator.poly_label(poly, label, color=colors(c, True))
                            if(c==1):
                                crop_image = save_one_box(poly, im_crop, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                                cv2.imshow('crop_image', crop_image)
                                two_stage_pred = predict_image_directly(crop_image, two_stage_model, 0)
                                print(f'two_stage_pred:{two_stage_pred}')
                                c = two_stage_pred
                                classnumber = torch.cat((classnumber, torch.tensor([c])))
                                class_indices = {
                                    0: "3L_0", 1: "3L_1", 2: "3L_10", 3: "3L_100", 4: "3L_101", 5: "3L_11", 6: "3L_12", 7: "3L_13",
                                    8: "3L_15", 9: "3L_16", 10: "3L_17", 11: "3L_18", 12: "3L_19", 13: "3L_2", 14: "3L_20", 15: "3L_21",
                                    16: "3L_22", 17: "3L_23", 18: "3L_25", 19: "3L_26", 20: "3L_27", 21: "3L_29", 22: "3L_3", 23: "3L_30",
                                    24: "3L_31", 25: "3L_4", 26: "3L_5", 27: "3L_6", 28: "3L_7", 29: "3L_8", 30: "3L_9", 31: "Alkaline_0",
                                    32: "Alkaline_1", 33: "Alkaline_10", 34: "Alkaline_100", 35: "Alkaline_101", 36: "Alkaline_102",
                                    37: "Alkaline_103", 38: "Alkaline_106", 39: "Alkaline_107", 40: "Alkaline_108", 41: "Alkaline_109",
                                    42: "Alkaline_11", 43: "Alkaline_110", 44: "Alkaline_111", 45: "Alkaline_112", 46: "Alkaline_113",
                                    47: "Alkaline_114", 48: "Alkaline_12", 49: "Alkaline_13", 50: "Alkaline_14", 51: "Alkaline_15",
                                    52: "Alkaline_16", 53: "Alkaline_17", 54: "Alkaline_18", 55: "Alkaline_19", 56: "Alkaline_2",
                                    57: "Alkaline_20", 58: "Alkaline_21", 59: "Alkaline_22", 60: "Alkaline_23", 61: "Alkaline_24",
                                    62: "Alkaline_25", 63: "Alkaline_26", 64: "Alkaline_27", 65: "Alkaline_28", 66: "Alkaline_29",
                                    67: "Alkaline_3", 68: "Alkaline_30", 69: "Alkaline_31", 70: "Alkaline_32", 71: "Alkaline_33",
                                    72: "Alkaline_34", 73: "Alkaline_35", 74: "Alkaline_36", 75: "Alkaline_37", 76: "Alkaline_4",
                                    77: "Alkaline_5", 78: "Alkaline_6", 79: "Alkaline_7", 80: "Alkaline_8", 81: "Alkaline_9", 82: "LiOnce_0",
                                    83: "LiOnce_1", 84: "LiOnce_10", 85: "LiOnce_2", 86: "LiOnce_3", 87: "LiOnce_4", 88: "LiOnce_5",
                                    89: "LiOnce_6", 90: "LiOnce_7", 91: "LiOnce_8", 92: "LiOnce_9", 93: "NICD_0", 94: "NICD_1",
                                    95: "NICD_2", 96: "NICD_3", 97: "NICD_4", 98: "NICD_5", 99: "NICD_6", 100: "NIMH_1", 101: "NIMH_100",
                                    102: "NIMH_101", 103: "NIMH_102", 104: "NIMH_103", 105: "NIMH_104", 106: "NIMH_105",
                                    107: "NIMH_106", 108: "NIMH_107", 109: "NIMH_108", 110: "NIMH_11", 111: "NIMH_111",
                                    112: "NIMH_112", 113: "NIMH_12", 114: "NIMH_14", 115: "NIMH_16", 116: "NIMH_17", 117: "NIMH_18",
                                    118: "NIMH_2", 119: "NIMH_20", 120: "NIMH_21", 121: "NIMH_22", 122: "NIMH_23", 123: "NIMH_24",
                                    124: "NIMH_25", 125: "NIMH_26", 126: "NIMH_28", 127: "NIMH_29", 128: "NIMH_3", 129: "NIMH_31",
                                    130: "NIMH_32", 131: "NIMH_33", 132: "NIMH_34", 133: "NIMH_35", 134: "NIMH_4", 135: "NIMH_6",
                                    136: "NIMH_7", 137: "NIMH_8", 138: "ZnMn_0", 139: "ZnMn_1", 140: "ZnMn_10", 141: "ZnMn_12",
                                    142: "ZnMn_13", 143: "ZnMn_14", 144: "ZnMn_15", 145: "ZnMn_16", 146: "ZnMn_17", 147: "ZnMn_18",
                                    148: "ZnMn_19", 149: "ZnMn_2", 150: "ZnMn_20", 151: "ZnMn_21", 152: "ZnMn_22", 153: "ZnMn_23",
                                    154: "ZnMn_24", 155: "ZnMn_25", 156: "ZnMn_3", 157: "ZnMn_4", 158: "ZnMn_5", 159: "ZnMn_6",
                                    160: "ZnMn_7", 161: "ZnMn_8", 162: "ZnMn_9"
                                }
                                batterynames = class_indices[c]
                                # label  = f'{battery_information[batterynames]} {maxconf:.2f}'
                                label  = f'{batterynames}'
                                annotator.poly_label(poly, label, color=colors(c, True))
                            elif(c==2):
                                label = f'{names[c]} {conf:.2f}'
                                annotator.poly_label(poly, label, color=colors(c, True))
                                classnumber = torch.cat((classnumber, torch.tensor([c+200])))
                            elif(c==3):
                                label = f'{names[c]} {conf:.2f}'
                                annotator.poly_label(poly, label, color=colors(c, True))
                                classnumber = torch.cat((classnumber, torch.tensor([c+200])))
                        else:
                            classnumber = det[:,-1]
                            label = f'{names[c]} {conf:.2f}'
                            annotator.poly_label(poly, label, color=colors(c, True))

                        if save_img or save_crop or view_img:  # Add poly to image
                            c = int(cls)  # integer class
                            # batterynames = names[c]
                            # if conf>0.5:
                            #     label = f'{batterynames} {conf:.2f}'
                            #     annotator.poly_label(poly, label, color=colors(c, True))
                            # else:
                            #     label = None
                            #     annotator.poly_label(poly, label, color=colors(c, True))
                            if save_crop: # Yolov5-obb doesn't support it yet
                                # save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                                pass
                    if grasp:
                        controller = RobotController()
                        controller.grasp_and_move(classnumber, new_center_x, new_center_y, angle_str,ChangeBase=True) 
            if no_battery:
                print("No battery detected!!!!!!!!!!!!!!!!!!!!!!")
                conveyor = send_start_signal(conveyor)
                time.sleep(2)
                break
            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({time.time() - t1:.3f}s)')

            # Stream results
            im0 = annotator.result()

            cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
            cv2.imshow('detect', im0)
            if cv2.waitKey(1)== ord('q') : # 1 millisecond
                raise StopIteration

            if grasp:    
                # controller = RobotController()
                while(True):
                    d = [450, -45, 480, 175, 3, 110]
                    pos =  controller.ask_item_demo()
                    if(pos == d):
                        break
                    else:
                        continue 
                # Save results (image with detections)
                # if save_img:
                #     if vid_path[i] != save_path:  # new video
                #         vid_path[i] = save_path
                #         if isinstance(vid_writer[i], cv2.VideoWriter):
                #             vid_writer[i].release()  # release previous video writer
                #         if vid_cap:  # video
                #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #         else:  # stream
                #             fps, w, h = 30, im0.shape[1], im0.shape[0]
                #             save_path += '.mp4'
                #         vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                #     vid_writer[i].write(im0)

        # # Print results
        # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        # if save_txt or save_img:
        #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        # if update:
        #     strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/yolov5n_DroneVehicle/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='/media/test/4d846cae-2315-4928-8d1b-ca6d3a61a3c6/DroneVehicle/val/raw/images/', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[840], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--two-stage', default=False, action='store_true', help='use two stage model')
    parser.add_argument('--grasp', default=False, action='store_true', help='grasp batttery')  
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)