import cv2
import numpy as np
import pyrealsense2 as rs
from cv2 import aruco

def convert_depth_to_phys_coord_using_realsense(x, y, depth, cameraInfo):
  _intrinsics = rs.intrinsics()
  _intrinsics.width = cameraInfo.width
  _intrinsics.height = cameraInfo.height
  _intrinsics.ppx = cameraInfo.ppx
  _intrinsics.ppy = cameraInfo.ppy
  _intrinsics.fx = cameraInfo.fx
  _intrinsics.fy = cameraInfo.fy

  _intrinsics.model  = rs.distortion.none
  _intrinsics.coeffs = [i for i in cameraInfo.coeffs]
  result = rs.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
  #result[0]: right, result[1]: down, result[2]: forward
  return result[1], result[0], result[2]

def get_calibration_data():
    cam_resolution = (1920, 1080)
    rs_cam = RS(cam_resolution[0], cam_resolution[1], frame_rate=30)
    detect_aruco = Aruco()
    return cam_resolution, rs_cam, detect_aruco

def calibrated_run(cam_resolution,rs_cam, detect_aruco):      
    with open('Aruco_info.txt', 'w') as file:
            rs_cam.update_raw_frames()
            color_frame = rs_cam.get_rgb_frame()
            display_frame = color_frame.copy()
            gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
            
            markerCorners, markerIds, rejectedImgPoints = aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)

            if len(markerCorners) != 0:
                detect_aruco.updateArucoInfo(markerIds, markerCorners)
                detect_aruco.drawMarkerCoord(markerIds, display_frame)
                # print(detect_aruco.mergeArucoInfo(markerIds, markerCorners)[0][0])

                # corners_precise = cv2.cornerSubPix(
                #     cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY),
                #     markerCorners,
                #     (11, 11),
                #     (-1, -1),
                #     criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                #     )

                origin = detect_aruco.origin
                x_point = detect_aruco.x
                y_point = detect_aruco.y

                origin_x, origin_y, origin_z = convert_depth_to_phys_coord_using_realsense(origin[0], origin[1], rs_cam.get_depth_value(int(origin[0]), int(origin[1])), rs_cam.intrinsics)
                xpoint_x, xpoint_y, xpoint_z = convert_depth_to_phys_coord_using_realsense(x_point[0], x_point[1], rs_cam.get_depth_value(int(x_point[0]), int(x_point[1])), rs_cam.intrinsics)
                ypoint_x, ypoint_y, ypoint_z = convert_depth_to_phys_coord_using_realsense(y_point[0], y_point[1], rs_cam.get_depth_value(int(y_point[0]), int(y_point[1])), rs_cam.intrinsics)
                new_coord_origin = np.array([origin_x, origin_y, origin_z])
                new_coord_x = np.array([xpoint_x, xpoint_y, xpoint_z])
                new_coord_y = np.array([ypoint_x, ypoint_y, ypoint_z])


                print("----------------------------------------")
                print(f"origin_x = {origin_x}")
                print(f"origin_y = {origin_y}")
                print(f"origin_z = {origin_z}")
                print("----------------------------------------")
                print("")

                print("----------------------------------------")
                print(f"xpoint_x = {xpoint_x}")
                print(f"xpoint_y = {xpoint_y}")
                print(f"xpoint_z = {xpoint_z}")
                print("----------------------------------------")
                print("")

                print("----------------------------------------")
                print(f"ypoint_x = {ypoint_x}")
                print(f"ypoint_y = {ypoint_y}")
                print(f"ypoint_z = {ypoint_z}")
                print("----------------------------------------")
                print("")

                                # 將數據寫入到文件中
                file.write(f"{origin_x},")
                file.write(f"{origin_y},")
                file.write(f"{origin_z},")
                file.write(f"{xpoint_x},")
                file.write(f"{xpoint_y},")
                file.write(f"{xpoint_z},")
                file.write(f"{ypoint_x},")
                file.write(f"{ypoint_y},")
                file.write(f"{ypoint_z}")

                # 绘制棋盘格角点
                # frame = cv2.drawChessboardCorners(frame, chessboard_size, corners_precise, ret)
                cv2.circle(color_frame, (int(origin[0]), int(origin[1])), 5, (0, 255, 0), -1)
                cv2.circle(color_frame, (int(x_point[0]), int(x_point[1])), 5, (0, 0, 255), -1)
                # cv2.circle(color_frame, (int(y_point[0]), int(y_point[1])), 5, (255, 0, 0), -1)


                # Calculate and display the distance between the corners
                distance = np.linalg.norm(new_coord_origin - new_coord_x)
                distance_text = f"Distance: {distance:.2f} m"
                cv2.putText(color_frame, distance_text, (int(origin[0]) + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.resize(color_frame, (640, 480))
    return color_frame

class RS:
    def __init__(self, rgb_resolution=(1920, 1080), depth_resolution=(1280, 720), frame_rate=30):
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, rgb_resolution[0], rgb_resolution[1], rs.format.bgr8, frame_rate)
        self.config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, frame_rate)
        self.pipeline = rs.pipeline()
        self.pipeline.start(self.config)
        self.pipeline.stop()
        prof =self.pipeline.start(self.config)

        s = prof.get_device().query_sensors()[1]
        s.set_option(rs.option.enable_auto_exposure, 0)
        s.set_option(rs.option.enable_auto_white_balance, 0)
        s.set_option(rs.option.backlight_compensation, 1)
    
        s.set_option(rs.option.exposure, 260)
        s.set_option(rs.option.gain, 64)
        s.set_option(rs.option.brightness, 0)
        s.set_option(rs.option.contrast, 50)
        s.set_option(rs.option.gamma, 300)
        s.set_option(rs.option.hue, 0)
        s.set_option(rs.option.saturation, 64)
        s.set_option(rs.option.sharpness, 50)
        s.set_option(rs.option.white_balance, 6000)
        
        self.align = rs.align(rs.stream.color)
        self.intrinsics = self.update_intrinsics()
        
    def update_raw_frames(self):
        while True:
            try:
                frames = self.pipeline.wait_for_frames()   
            except:
                print("Frame didn't arrive within certain time, try again.")
                continue
            
            aligned_frames = self.align.process(frames)

            # Get aligned RGB and depth frames
            self.color_frame_raw = aligned_frames.get_color_frame()
            self.depth_frame_raw = aligned_frames.get_depth_frame()

            if not self.color_frame_raw or not self.depth_frame_raw:
                continue
            else:
                return True
            
    def update_intrinsics(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        return intrinsics

    def get_depth_frame(self):
        # Get depth frame
        self.depth_frame = np.asanyarray(self.depth_frame_raw.get_data())
        return self.depth_frame
    
    def get_rgb_frame(self):
        # Get RGB frame
        self.color_frame = np.asanyarray(self.color_frame_raw.get_data())
        return self.color_frame

    def get_depth_value(self, x, y):
        try:
            depth_values = np.array(self.depth_frame_raw.get_distance(x, y))
        except:
            depth_values = None
            print("Error: Could not get depth values.")
        return depth_values
    
    def stop(self):
        self.pipeline.stop()

class Aruco:
    def __init__(self):
        self.id = 0
        self.origin = None
        self.x = None  
        self.y = None

    def updateArucoInfo(self, ids, corners):
        self.id = ids
        self.origin = np.array(corners[0][0][0], dtype=int)
        self.x = np.array(corners[0][0][3], dtype=int)
        self.y = np.array(corners[0][0][1], dtype=int)

    def drawMarkerCoord(self, id, image):
        a = []
        if id[0][0] == 0:
            cv2.line(image, self.origin, self.x, color=(0, 255, 0), thickness=3)
            cv2.line(image, self.origin, self.y, color=(0, 0, 255), thickness=3)
            cv2.circle(image, tuple(self.origin), 5, (255, 0, 0), -1)
            cv2.circle(image, tuple(self.x), 5, (0, 255, 0), -1)
            cv2.circle(image, tuple(self.y), 5, (0, 0, 255), -1)
            
    def getMarkerCnt(id, corner, image):
        markerCnt = []
        for i in range(len(id)):
            c = corner[i][0]
            markerCnt.append([np.mean(c[:, 0], dtype=int),np.mean(c[:, 1], dtype=int)])
        return markerCnt 

    def mergeArucoInfo(ids, corners):
        aruco_dict = {}

        if ids is not None:
            for i in range(len(ids)):
                aruco_dict[ids[i][0]] = corners[i][0].tolist()
        else:
            print("Can't detect aruco to merge!!!")

        return aruco_dict

# Aruco dictionary initialize
aruco_dict =  aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

if __name__ == "__main__":
    cam_resolution, rs_cam, detect_aruco = get_calibration_data()
    #while True:
    calibrated_run(cam_resolution,rs_cam, detect_aruco)

    #cv2.destroyAllWindows()