#!/usr/bin/env python
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from collections import defaultdict
import math
import pyrealsense2 as rs

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

sys.path.append('/home/yang/jojo/battery_project/battery_detect_ros/src/battery_detect_withGUI/src/yolov5_obb/patchmix')
sys.path.append('/home/yang/jojo/battery_project/battery_detect_ros/src/battery_detect_withGUI/src/yolov5_obb/tools')
import torch.nn as nn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from patchmix.config.finetune.vit_base_finetune import vit_base_finetune
from patchmix.config.finetune.vit_small_finetune import vit_small_finetune
from patchmix.config.finetune.vit_tiny_finetune import vit_tiny_finetune
from patchmix.module.vits import ViT
from PIL import Image
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from tools.RobotArm import RobotController

# import rospy
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge

class YOLODetector(QThread):
    update_image_signal = pyqtSignal(np.ndarray)
    def __init__(self,
                 rs_cam=None, 
                 weights='/home/yang/jojo/battery_project/battery_detect_ros/src/battery_detect_withGUI/src/yolov5_obb/runs/train/shape0305/weights/best.pt', 
                 source='4', 
                 imgsz=(1080, 1080), 
                 conf_thres=0.5, 
                 iou_thres=0.2, 
                 max_det=1000, 
                 device='0', 
                 view_img=True, 
                 save_txt=False, 
                 save_conf=False, 
                 save_crop=False, 
                 nosave=False, 
                 classes=None, 
                 agnostic_nms=True, 
                 augment=False, 
                 visualize=False, 
                 update=False, 
                 project='runs/detect', 
                 name='exp', 
                 exist_ok=False, 
                 line_thickness=3, 
                 hide_labels=False, 
                 hide_conf=False, 
                 half=False, 
                 dnn=False, 
                 two_stage=True, 
                 grasp=False):
        super().__init__()
        print('rs_cam type',type(rs_cam))
        print('rs_cam ',rs_cam)
        self.weights = weights
        self.source = source
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = select_device(device)
        self.view_img = view_img
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.save_crop = save_crop
        self.nosave = nosave
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.update = update
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half
        self.dnn = dnn
        self.two_stage = two_stage
        self.grasp = grasp
        self.csv_file_path = '/home/yang/jojo/battery_project/battery_detect_ros/src/battery_detect_withGUI/src/yolov5_obb/coculate_data.csv'


        # 初始化資料
        data = {
            'BatteryType': ['circle','3L', 'Alkaline','LiOnce', 'NICD', 'NIMH','ZnMn','square','package'],
            'Count': [0,0,0,0,0,0,0,0,0]
        }

        # 創建 DataFrame
        df = pd.DataFrame(data)

        # 寫入 CSV 檔案
        df.to_csv(self.csv_file_path, index=False)
        print(f"成功初始化 CSV 檔案：'{self.csv_file_path}'")
        
        self.model = DetectMultiBackend(self.weights, device=self.device,dnn=self.dnn)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        
        if self.two_stage:
            self.two_stage_model = ViT(patch_size=16, img_size=224, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, drop_path_rate=0.1)
            self.two_stage_model.head = nn.Linear(self.two_stage_model.head.in_features, 163)
            self.two_stage_model.cuda(0)
            two_stage_weights = '/home/yang/jojo/battery_project/battery_detect_ros/src/battery_detect_withGUI/src/yolov5_obb/patchmix/out/vit-small_imagenet1k_epochs_100_lr_0.0005_battery-163/best.pth'
            self.two_stage_model = torch.load(two_stage_weights, map_location=torch.device(0))
            print("=> loaded two stage model '{}'".format(two_stage_weights))
        
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        self.half &= (self.pt or self.jit or self.engine) and self.device.type != 'cpu'
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
        self.model.warmup(imgsz=(1, 3, *self.imgsz), half=self.half)
        
        self.dt, self.seen = [0.0, 0.0, 0.0], 0
        self.cam_resolution = (1920, 1080)
        self.rs_cam = rs_cam
        self.conveyor = False
        #self.controller = RobotController()
        self.state=True
        
        with open('/home/yang/jojo/battery_project/battery_detect_ros/src/battery_detect_withGUI/src/battery_GUI/Aruco_info.txt', 'r') as file:
            # 逐行讀取文件中的內容
            for line in file:
                # 將每行按逗號分隔成列表，並將每個值轉換為浮點數
                values = [float(val) for val in line.strip().split(',')]
                #print("Values:", values)
                # 將每個值分配給相應的變量
                origin_x, origin_y, origin_z = values[:3]
                xpoint_x, xpoint_y, xpoint_z = values[3:6]
                ypoint_x, ypoint_y, ypoint_z = values[6:]
        
                # Define new origin point, x and y by Aruco marker
                # 數值是經由 realsense.py 取得 Aruco Marker 指定的原點，X跟Y (以realsense 的世界座標呈現)
                self.new_origin = np.array([origin_x, origin_y, origin_z]) *1000 
                self.new_x_point = np.array([xpoint_x, xpoint_y, xpoint_z]) *1000
                self.new_y_point = np.array([ypoint_x, ypoint_y,ypoint_z]) *1000
        # ROS setup
        # rospy.init_node('yolo_detector_node', anonymous=True)
        # self.bridge = CvBridge()
        # self.image_pub = rospy.Publisher('/yolo_detected_image', Image, queue_size=10)


    def predict_image_directly(self, image):
        resize_im = 224 > 32
        transform_list = []
        if resize_im:
            size = int((256 / 224) * 224)
            transform_list.append(transforms.Resize(size, interpolation=3))
            transform_list.append(transforms.CenterCrop(224))

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        transform = transforms.Compose(transform_list)

        image = image[:, :, [2, 1, 0]]
        image = Image.fromarray(image)

        transformed_image = transform(image).unsqueeze(0)
        transformed_image = transformed_image.to(self.device)

        self.two_stage_model.eval()
        with torch.no_grad():
            output = self.two_stage_model(transformed_image)
            _, pred = output.topk(1, 1, True, True)
            pred = pred.item()
        return pred
    
    def check_camera_connection(self):
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("No RealSense cameras found.")
            return False
        else:
            print("RealSense camera connected.")
            for i, dev in enumerate(devices):
                print(f"Device {i+1}: {dev.get_info(rs.camera_info.name)}")
            return True
    
    def send_start_signal(self):
        print("start the conveyor belt")
        self.conveyor = True

    def send_stop_signal(self):
        print("stop the conveyor belt")
        self.conveyor = False
        return self.conveyor
        
    @torch.no_grad()
    def run(self):
        self.conveyor = self.send_start_signal()

        while (self.state):
            empty_count = 0
            count = 0
            while self.conveyor:
                print("conveyor is running")
                self.rs_cam.update_raw_frames()
                color_frame = self.rs_cam.get_rgb_frame()
                hsv_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
                hue, saturation, value = cv2.split(hsv_frame)
                zero_channel = np.zeros_like(hue) * 100
                merge_image = cv2.merge([hue, zero_channel, value])
                imgs = [None]
                im0s = [None]
                sources = [self.source]
                im0s[0] = color_frame
                imgs[0] = merge_image
                path = sources
                img = [letterbox(x, new_shape=self.imgsz, auto=self.pt)[0] for x in imgs]
                img = np.stack(img, 0)
                img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
                img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)
                img /= 255.0

                img = torch.from_numpy(img).to(self.device)
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                pred = self.model(img, augment=self.augment, visualize=self.visualize)
                pred = non_max_suppression_obb(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, multi_label=True, max_det=self.max_det)
                for i, det in enumerate(pred):
                    im0 = im0s[i].copy()
                    pred_poly = rbox2poly(det[:, :5])
                    pred_poly = scale_polys(img.shape[2:], pred_poly, im0.shape)
                    annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
                    det = torch.cat((pred_poly, det[:, -2:]), dim=1)
                    for *poly, conf, cls in reversed(det):
                        c = int(cls)
                        label = f'{self.names[c]} {conf:.2f}'
                        annotator.poly_label(poly, label, color=colors(c, True))
                    for battery in pred_poly:
                        for i in range(0, 8, 2):
                            x, _ = battery[i], battery[i + 1]
                            if x < 350:
                                self.conveyor = self.send_stop_signal()
                                break
            time.sleep(2)
            break_out = False
            while (self.state):
                s = 0.0
                if break_out:
                    break
                t0 = time.time()
                self.rs_cam.update_raw_frames()
                color_frame = self.rs_cam.get_rgb_frame()
                hsv_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
                hue, saturation, value = cv2.split(hsv_frame)
                zero_channel = np.zeros_like(hue) * 100
                merge_image = cv2.merge([hue, zero_channel, value])
                imgs = [None]
                im0s = [None]
                sources = [self.source]
                im0s[0] = color_frame
                imgs[0] = merge_image
                path = sources
                img = [letterbox(x, new_shape=self.imgsz, auto=self.pt)[0] for x in imgs]
                img = np.stack(img, 0)
                img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
                img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)
                img /= 255.0

                img = torch.from_numpy(img).to(self.device)
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                t1 = time_sync()
                pred = self.model(img, augment=self.augment, visualize=self.visualize)
                pred = non_max_suppression_obb(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, multi_label=True, max_det=self.max_det)
                battery_tensor = pred[0]
                if battery_tensor.numel() == 0:
                    empty_count += 1
                    if empty_count <= 50:
                        continue
                    else:
                        print("No battery detected")
                        self.conveyor = self.send_start_signal()
                        time.sleep(2)
                        break_out = True
                        break
                t2 = time_sync()
                no_battery = False
                for i, det in enumerate(pred):
                    pred_poly = rbox2poly(det[:, :5])
                    self.seen += 1
                    p, s, im0 = path[i], '%g: ', im0s[i].copy()
                    s += f'{i}: '
                    p = Path(p)
                    # save_path = str(self.save_dir / p.name)
                    # txt_path = str(self.save_dir / 'labels' / p.stem)
                    s += '%gx%g ' % img.shape[2:]
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                    imc = im0.copy() if self.save_crop else im0
                    im_crop = im0.copy()
                    annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
                    if len(det):
                        coculate_data_count=0
                        pred_poly = scale_polys(img.shape[2:], pred_poly, im0.shape)
                        pred_length = torch.tensor([])
                        
                        for i, poly in enumerate(pred_poly):
                            polys = poly.reshape(4, 2).cpu().numpy()
                            calculator = PolygonCalculator()
                            length = calculator.calculate_length(polys)
                            pred_length = torch.cat((pred_length, torch.tensor([length])))
                            ratio, longp1, longp2 = calculator.calculate_ratio(polys)
                            centerx, centery = calculator.calculate_center(polys)
                            new_center_x, new_center_y, new_center_z = convert_depth_to_phys_coord_using_realsense(centerx, centery, self.rs_cam.get_depth_value(int(centerx), int(centery)), self.rs_cam.intrinsics)
                            
                            print(f"origin_center:{self.new_origin[0], self.new_origin[1]}")
                            #print(f"new_center: {new_center_x, new_center_y}")
                            
                            battery_center = np.array(convert_depth_to_phys_coord_using_realsense(centerx, centery, self.rs_cam.get_depth_value(int(centerx), int(centery)), self.rs_cam.intrinsics))[:2] *1000

                            new_x_axis = np.array(self.new_x_point) - np.array(self.new_origin)
                            new_x_axis = new_x_axis[:2]
                            new_y_axis = np.array(self.new_y_point) - np.array(self.new_origin)
                            new_y_axis = new_y_axis[:2]

                            new_x_axis_basis = new_x_axis / np.linalg.norm(new_x_axis)
                            new_y_axis_basis = new_y_axis / np.linalg.norm(new_y_axis)

                            M_newCoordBasis = np.stack((new_x_axis_basis, new_y_axis_basis)).T                      # Calculate matrix M_newCoordBasis's inverse
                            M_newCoordBasis_inv = np.linalg.inv(M_newCoordBasis)                                    # Calculate obj_vec's projection on new coordinate system
                            

                            
                            obj_vec_newCoord = np.around(np.dot(M_newCoordBasis_inv, battery_center), decimals=2)
                            Aruco_Orign_vec_newCoord = np.around(np.dot(M_newCoordBasis_inv, self.new_origin[0:2]), decimals=2)
                            print("Battery Center vector in new coordinate system is {}y_component_vec".format(obj_vec_newCoord - Aruco_Orign_vec_newCoord))
                            distance = obj_vec_newCoord - Aruco_Orign_vec_newCoord
                            new_center_x = distance[0]
                            new_center_y = distance[1]
                            
                            
                            # new_center_x = (new_center_x - self.new_origin[0]) * 1000
                            # new_center_y = (new_center_y - self.new_origin[1]) * 1000
                            # new_center_z = new_center_z - self.new_origin[2]
                            
                            longp1x, longp1y = longp1
                            longp2x, longp2y = longp2
                            longp1_tfx, longp1_tfy = calculator.calibration(longp1x, longp1y)
                            longp2_tfx, longp2_tfy = calculator.calibration(longp2x, longp2y)
                            angle = calculator.calculate_angle(float(longp1_tfx), float(longp1_tfy), float(longp2_tfx), float(longp2_tfy))
                            angle_str = str(angle)
                            print(f"Poligon {i+1}:")
                            print(f" pred_poly: {pred_poly[i]}")
                            print(f" center: {centerx, centery}")

                        det = torch.cat((pred_poly, det[:, -2:]), dim=1)
                        condition1 = pred_length <= 250
                        det = det[condition1]
                        last_column = det[:, -1]
                        condition2 = last_column != 0
                        det = det[condition2]
                        #print(f"det: {det}")
                        if det.numel() == 0:
                            count += 1
                            if count <= 50:
                                continue
                            else:
                                print("No battery detected")
                                no_battery = True
                                break
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()
                            #print(int(c))
                            s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
                        classnumber = torch.tensor([])

                        for *poly, conf, cls in reversed(det):
                            c = int(cls)
                            poly_tensor = torch.stack(poly)
                            print(f"--------------poly_tensor: {poly_tensor}")
                            if self.two_stage:
                                if c == 0:
                                    label = f'{self.names[c]} {conf:.2f}'
                                    annotator.poly_label(poly, label, color=colors(c, True))
                                    coculate_data_count=0
                                if c == 1:
                                    crop_image = save_one_box(poly, im_crop, BGR=True)
                                    # cv2.imshow('crop_image', crop_image)
                                    two_stage_pred = self.predict_image_directly(crop_image)
                                    print(f'two_stage_pred: {two_stage_pred}')
                                    c = two_stage_pred
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
                                    #3L
                                    if c >= 0 and c <= 30:
                                        coculate_data_count=1
                                    #Alkaline
                                    elif c >= 31 and c <= 81:
                                        coculate_data_count=2
                                    #LiOnce
                                    elif c >= 82 and c <= 92:
                                        coculate_data_count=3
                                    #NICD
                                    elif c >= 93 and c <= 99:
                                        coculate_data_count=4
                                    #NIMH
                                    elif c >= 100 and c <= 137:
                                        coculate_data_count=5
                                    #ZnMn
                                    elif c >= 138 and c <= 162:
                                        coculate_data_count=6
                                    
                                    classnumber = torch.cat((classnumber, torch.tensor([c])))
                                    batterynames = class_indices[c]
                                    label  = f'{batterynames}'
                                    annotator.poly_label(poly, label, color=colors(c, True))
                                elif c == 2:
                                    coculate_data_count=7
                                    label = f'{self.names[c]} {conf:.2f}'
                                    annotator.poly_label(poly, label, color=colors(c, True))
                                    classnumber = torch.cat((classnumber, torch.tensor([c + 200])))
                                elif c == 3:
                                    coculate_data_count=8
                                    label = f'{self.names[c]} {conf:.2f}'
                                    annotator.poly_label(poly, label, color=colors(c, True))
                                    classnumber = torch.cat((classnumber, torch.tensor([c + 200])))
                            else:
                                classnumber = det[:, -1]
                                label = f'{self.names[c]} {conf:.2f}'
                                annotator.poly_label(poly, label, color=colors(c, True))
                                
 
                    # 讀取 CSV 檔案
                    df = pd.read_csv(self.csv_file_path)
                    df.at[coculate_data_count, 'Count'] += 1    
                    df.to_csv(self.csv_file_path, index=False)

                    if self.grasp:      
                        self.controller.grasp_and_move(classnumber, new_center_x, new_center_y, angle_str, ChangeBase=True)

                LOGGER.info(f'{s}Done. ({time.time() - t1:.3f}s)')
                im0 = annotator.result()
                # update_callback(im0)
                self.update_image_signal.emit(im0)
                # if cv2.waitKey(1) == ord('q'):
                #     raise StopIteration
                if self.grasp:
                    while True:
                        d = [450, -45, 480, 175, 3, 110]
                        pos = self.controller.ask_item_demo()
                        if pos == d:
                            break
                        else:
                            continue

                if no_battery:
                    print("No battery detected!!!!!!!!!!!!!!!!!!!!!!")
                    self.conveyor = self.send_start_signal()
                    time.sleep(2)
                    break

    def close(self):
        self.state=False
        if self.grasp:
            self.controller.emergency_stop()
        self.rs_cam.pipeline.stop()
    
