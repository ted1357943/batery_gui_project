import cv2 as cv
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel
import numpy as np
import pyrealsense2 as rs

class QT_RealsenseThread(QThread):
    # Create image_signal to emit image
    image_signal = pyqtSignal(np.ndarray)

    def __init__(self, resolution=(1920, 1080), frame_rate=30):
        super().__init__()

        self.opened = False

        # Initialize the RealSense pipeline
        self.pipeline = rs.pipeline()

        # self.rs_configure the pipeline for depth and color streams
        self.rs_config = rs.config()
        self.rs_config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, frame_rate)
        
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

    def run(self):
        try:
            self.pipeline.start(self.rs_config)
            self.opened = True
        except RuntimeError:
            self.image_signal.emit(np.array(None))
            return

        while self.opened:
            try:
                frames = self.pipeline.wait_for_frames()
            except RuntimeError:
                self.image_signal.emit(np.array(None))
                self.opened = False
                break
            
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            self.image_signal.emit(color_image)
            

    def close(self):
        if self.opened:
            self.pipeline.stop()
            self.opened = False