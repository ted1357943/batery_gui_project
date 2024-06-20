import argparse
import math
import os
import time
from typing import Iterable, Optional

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from timm.data import Mixup, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, ModelEma, accuracy
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from config.finetune.vit_base_finetune import vit_base_finetune
from config.finetune.vit_small_finetune import vit_small_finetune
from config.finetune.vit_tiny_finetune import vit_tiny_finetune
from module.vits import ViT
from utils import misc
from utils.logger import Logger, console_logger
from utils.misc import AverageMeter
import cv2
from PIL import Image

def predict_image_directly(image_path, model, device, args):
    # Define transformation for evaluation
    resize_im = args.input_size > 32
    transform_list = []
    if resize_im:
        size = int((256 / 224) * args.input_size)  # Adjust the resizing factor
        transform_list.append(transforms.Resize(size, interpolation=3))
        transform_list.append(transforms.CenterCrop(args.input_size))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    transform = transforms.Compose(transform_list)

    # Transform the image
    image = Image.open(image_path).convert('RGB')
    transformed_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move image to the device
    transformed_image = transformed_image.to(device)

    # Set model to evaluation mode
    model.eval()

    # Disable gradient calculation
    with torch.no_grad():
        output = model(transformed_image)
        _, pred = output.topk(1, 1, True, True)
    
    return pred.item()

def extract_class_from_path(image_path):
    # 使用 os.path.dirname 獲取包含文件的目錄的路徑
    directory_path = os.path.dirname(image_path)
    # 使用 os.path.basename 獲取目錄路徑的最後一部分，即類別名稱
    class_name = os.path.basename(directory_path)
    return class_name


def main(args):
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
    args.rank = args.rank
    num_class = 163
    cudnn.benchmark = True

    if args.arch == 'vit-tiny':
        model = ViT(patch_size=args.patch_size, img_size=args.input_size,
                    embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, drop_path_rate=args.drop_path)
    elif args.arch == 'vit-small':
        model = ViT(patch_size=args.patch_size, img_size=args.input_size,
                    embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, drop_path_rate=args.drop_path)
    elif args.arch == 'vit-base':
        model = ViT(patch_size=args.patch_size, img_size=args.input_size,
                    embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, drop_path_rate=args.drop_path)

    model.head = nn.Linear(model.head.in_features, num_class)

    model.cuda(args.rank)

    if args.evaluate:
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}'".format(args.evaluate))
            model = torch.load(
                args.evaluate, map_location=torch.device(args.rank))
            print("=> loaded pre-trained model '{}'".format(args.evaluate))
        else:
            print("=> no checkpoint found at '{}'".format(args.evaluate))
        image_path = '/home/iris/Documents/battery_inside/cylinder0315_blackground_cropv2/test/NIMH_2/battery_0_jpg.rf.b44b33f561a038066e6b07d227df0fa2.jpg'
        class_name  = extract_class_from_path(image_path)
        battery_class = predict_image_directly(image_path, model, 0, args)
        print('actual_class :'+ str(class_name)+' ; predict_class :' + str(class_indices[battery_class]))
        return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default='vit-small',
                        choices=['vit-tiny', 'vit-small', 'vit-base'])
    parser.add_argument("--pretrained-weights", type=str,
                        default='')
    parser.add_argument("--evaluate", type=str, default=None)
    return parser


if __name__ == '__main__':
    parser = parse_args()
    _args = parser.parse_args()

    if _args.arch == 'vit-tiny':
        args = vit_tiny_finetune()
    elif _args.arch == 'vit-small':
        args = vit_small_finetune()
    elif _args.arch == 'vit-base':
        args = vit_base_finetune()
    args.pretrained_weights = _args.pretrained_weights
    args.evaluate = _args.evaluate
    main(args)
