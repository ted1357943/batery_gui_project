import numpy as np
import math
import random
import torch

class PolygonCalculator:
    def __init__(self):
        pass
    
    def distance_2d(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def calculate_length(self, poly):
        width = self.distance_2d(poly[0], poly[1])
        length = self.distance_2d(poly[1], poly[2])
        return min(width, length)
            
    def calculate_ratio(self, poly):
        width = self.distance_2d(poly[0], poly[1])
        length = self.distance_2d(poly[1], poly[2])
        if width > length:
            ratio = width / length
            longp1, longp2 = poly[0], poly[1]
        else:
            ratio = length / width
            longp1, longp2 = poly[1], poly[2]
        return ratio, longp1, longp2

    def calculate_center(self, poly):
        x1, y1 = poly[0][0], poly[0][1]  # 右上
        x2, y2 = poly[1][0], poly[1][1]  # 左上
        x3, y3 = poly[2][0], poly[2][1]  # 左下
        x4, y4 = poly[3][0], poly[3][1]  # 右下
        centerx = (x1 + x2 + x3 + x4) / 4
        centery = (y1 + y2 + y3 + y4) / 4
        center = (centerx, centery)
        return center

    def calculate_angle(self, x1, y1, x2, y2):
        angle = math.atan((y2 - y1) / (x2 - x1))
        angle_degrees = math.degrees(angle)
        angle_degrees += 90
        return angle_degrees

    def calibration(self, centerx, centery):
        x1 = np.array([[333.7,350.32,1],[274.2,387,1],[175.66,375.82,1],[226.94,309.66,1],[282.14,243.14,1],[412.64,389.45,1],[182.88,234.02,1],[290.93,129.45,1],[193.93,115.33,1]])  
        y1 = np.array([[647.35],[696.09],[709.83],[643.80],[564.17],[652.55],[599.35],[459.57],[473.97]])  
        y2 = np.array([[419.51],[376.58],[282.51],[315.19],[345.6],[512.61],[240.62],[326.89],[209.2]]) 
        A = np.linalg.lstsq((x1), (y1), rcond=None)[0]
        B = np.linalg.lstsq((x1), (y2), rcond=None)[0]
        input_image_point = np.array([centerx, centery, 1])
        output_tcp_pointx = np.dot(input_image_point, A)
        output_tcp_pointy = np.dot(input_image_point, B)
        X = np.squeeze(output_tcp_pointx)
        Y = np.squeeze(output_tcp_pointy)
        return X, Y

    def uniform_sampling_in_box(self, poly, num_points):
        sampled_points = []
        x1, y1 = poly[0][0], poly[0][1]
        x2, y2 = poly[1][0], poly[1][1]
        x3, y3 = poly[2][0], poly[2][1]
        x4, y4 = poly[3][0], poly[3][1]
        for _ in range(num_points):
            s = random.uniform(0, 1)
            t = random.uniform(0, 1)
            x = (1 - s) * (1 - t) * x1 + s * (1 - t) * x2 + s * t * x3 + (1 - s) * t * x4
            y = (1 - s) * (1 - t) * y1 + s * (1 - t) * y2 + s * t * y3 + (1 - s) * t * y4
            sampled_points.append((int(x), int(y)))
        return sampled_points

    def count_points_in_region(self, region, points):
        count = 0
        for point in points:
            if point[0] >= region[0] and point[0] <= region[2] and point[1] >= region[1] and point[1] <= region[3]:
                count += 1
        return count
    def calculate_areas(self, poly):
        if isinstance(poly, np.ndarray):  # 检查 poly 是否是 NumPy 数组
            poly = torch.tensor(poly)      # 如果是，将其转换为 PyTorch 张量
        if poly.dim() == 1:
            poly = poly.view(-1, 8)

        # 重新排列座標以方便計算
        x1, y1, x2, y2, x3, y3, x4, y4 = poly[0][0], poly[0][1], poly[1][0], poly[1][1], poly[2][0], poly[2][1], poly[3][0], poly[3][1]
        
        # 計算兩個三角形的面積
        area1 = torch.abs((x2-x1) * (y3-y1) - (x3-x1) * (y2-y1)) / 2
        area2 = torch.abs((x3-x1) * (y4-y1) - (x4-x1) * (y3-y1)) / 2
        
        # 總面積
        total_area = area1 + area2
    
        return total_area.item()
