#!/usr/bin/env python
import os
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pandas as pd
import time
import cv2 as cv
import subprocess
import csv
import sys
sys.path.append('/home/yang/jojo/battery_project/battery_detect_ros/src/battery_detect_withGUI/src/yolov5_obb')
sys.path.append('/home/yang/jojo/battery_project/battery_detect_ros/src/battery_detect_withGUI/src/yolov5_obb/camera')
from realsense import calibrated_run
from realsense import get_calibration_data
from QT_realsense import QT_RealsenseCaliThread
from  realsense import RS

from Custom_Widgets.Widgets import *
from tools.RobotArm import RobotController
# Import the project directory
final_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.append(final_path)

# Set QT icon path (In Resource dir)
resource_path = os.path.abspath(os.path.join(final_path, "Resource"))
sys.path.append(resource_path)

# Set Data path
data_path = os.path.abspath(os.path.join(final_path, "Data"))


from View.bubble_detect_interface import *

from Model.openRS import QT_RealsenseThread
from Model.bubble_detect import preprocess as m_bubble_proc
from detect import YOLODetector

# from Model.bubble_detect import preprocess as m_bubble_proc
# from Model.FTP import ftp_client as ftpClient

# Main Class
class PyQt_MVC_Main(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # ---------- Setting Window size to full screen ---------- #
        # Get the available geometry of the primary screen
        screen_geometry = QApplication.primaryScreen().availableGeometry()

        # Set the height of the main window to the available height while keeping the width intact
        self.setGeometry(self.geometry().x(), self.geometry().y(), screen_geometry.width(), screen_geometry.height())
        # Get the width of the menu bar
        menu_bar_height = self.menuBar().size().height()

        # ---------- Apply JSON  ---------- #
        loadJsonStyle(self, self.ui)

        # ---------- Set Window icon  ---------- #
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(resource_path+"/self_made/Blob.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.setWindowIcon(icon)

        # ---------- Setting bubble detect class  ---------- #
        self.DetectProcess = m_bubble_proc.preprocess()
        self.Bubble = m_bubble_proc.bubble()

        # ---------- Show UI  ---------- #
        # Set the position of the window to (x, y)
        # Get the screen geometry
        launcher_width = QApplication.desktop().availableGeometry().x()
        launcher_height = QApplication.desktop().availableGeometry().y()
        self.move(launcher_width, launcher_height + menu_bar_height)
        self.show()
        self.csv_file_path = '/home/yang/jojo/battery_project/battery_detect_ros/src/battery_detect_withGUI/src/yolov5_obb/coculate_data.csv'
        # Set image label's size
        # self.ui.cameraLabel.setFixedSize(self.ui.cameraLabel.width(), self.ui.cameraLabel.height() - 10)
        # self.ui.calibratedLabel.setFixedSize(self.ui.cameraLabel.width(), self.ui.cameraLabel.height() - 10)
        # self.ui.bubbleLabel.setFixedSize(self.ui.cameraLabel.width(), self.ui.cameraLabel.height() - 10)

        # self.DetectProcess = m_bubble_proc.preprocess()
        # self.Bubble = m_bubble_proc.bubble()
        self.BubbleImgResolution = (1920, 1080)
        self.InputImgResolution = (1920, 1080)
        self.RS_Btn_State = -1
        self.RS_Cam_thread = None
        self.RS_Cam_Cali_thread = None
        self.RS_Cam_battery_thread = None
        ############### Tab: FTP ###############
        self.fn = None
        self.ftpServerIP = None
        self.ftpPort = 21 # FTP port, don't change
        self.ftpUserName = None
        self.ftpPassword = None
        self.ftpClient = []

        ########################################
        #############  Link Event  #############
        ########################################
        self.ui.cameraBtn.clicked.connect(lambda: self.cameraBtnClickEvent())
        # Bubble Detect
        self.ui.calibratedBtn.clicked.connect(lambda: self.calibratedBtnClickEvent())
        self.ui.bubbleBtn.clicked.connect(lambda: self.batteryBtnClickEvent())
        self.ui.stopBtn.clicked.connect(lambda: self.stopBtnClickEvent())
        self.ui.emergencyBtn.clicked.connect(lambda: self.emergencyBtnClickEvent())

        # button state
        self.cameraBtn_status = False
        self.calibratedBtn_status = False
        self.batteryBtn_status = False
        self.stopBtn_status = False
        self.emergencyBtn_status = False
        # Initialize button styles
        self.updateButtonStyles()
        
        #self.controller = RobotController()
        # 初始化計時器
        self.calibration_timer = QTimer(self)
        self.calibration_timer.timeout.connect(self.stopCalibration)
    ########################################
    ###############   Model  ###############
    ########################################
    def updateButtonStyles(self):
        # Define styles for active and inactive buttons
        active_style = '''background-color:#ff0; color:#f00;'''
        inactive_style = ''''''

        # Update button styles based on their statuses
        self.ui.cameraBtn.setStyleSheet(active_style if self.cameraBtn_status else inactive_style)
        self.ui.calibratedBtn.setStyleSheet(active_style if self.calibratedBtn_status else inactive_style)
        self.ui.bubbleBtn.setStyleSheet(active_style if self.batteryBtn_status else inactive_style)
        self.ui.stopBtn.setStyleSheet(active_style if self.stopBtn_status else inactive_style)
        self.ui.emergencyBtn.setStyleSheet(active_style if self.emergencyBtn_status else inactive_style)
    

    def cameraBtnClickEvent(self):
        self.close_RS_Cam()
        self.cameraBtn_status = True
        self.calibratedBtn_status = False
        self.batteryBtn_status = False
        self.stopBtn_status = False
        self.emergencyBtn_status = False
        self.updateButtonStyles()
        self.open_RS_Cam()

    def calibratedBtnClickEvent(self):
        self.close_RS_Cam()
        self.cameraBtn_status = False
        self.calibratedBtn_status = True
        self.batteryBtn_status = False
        self.stopBtn_status = False
        self.emergencyBtn_status = False
        self.ShowRobotCalibration()
        #self.calibrated_coordinate()
        self.updateButtonStyles()


    def batteryBtnClickEvent(self):
        self.close_RS_Cam()
        self.cameraBtn_status = False
        self.calibratedBtn_status = False
        self.batteryBtn_status = True
        self.stopBtn_status = False
        self.emergencyBtn_status = False
        self.battery_process()
        self.updateButtonStyles()

    def stopBtnClickEvent(self):
        self.close_RS_Cam()
        self.cameraBtn_status = False
        self.calibratedBtn_status = False
        self.batteryBtn_status = False
        self.stopBtn_status = True
        self.emergencyBtn_status = False
        self.show_stopImage()
        #self.controller.go_home()
        self.display_battery_grasp_info()
        self.updateButtonStyles() 
    
    def emergencyBtnClickEvent(self):
        self.emergency_RS_Cam()
        self.cameraBtn_status = False
        self.calibratedBtn_status = False
        self.batteryBtn_status = False
        self.stopBtn_status = False
        self.emergencyBtn_status = True
        self.show_emergencyImage()
        #self.controller.emergency_stop()
        self.updateButtonStyles()
        self.ShowEmergencyDialog()
       
    def HandleEmergencyDialogButtonClick(self):
        #Re-enable all buttons
        self.ui.cameraBtn.setDisabled(False)
        self.ui.calibratedBtn.setDisabled(False)
        self.ui.bubbleBtn.setDisabled(False)
        self.ui.emergencyBtn.setDisabled(False) 
        self.ui.stopBtn.setDisabled(False) 
    

    ###############   Realsense  ###############
    def open_RS_Cam(self):
        if self.RS_Cam_thread is None  or not self.RS_Cam_thread.isRunning():
            self.RS_Cam_thread = QT_RealsenseThread()
            if self.RS_Cam_thread.check_camera_connection():
                self.RS_Cam_thread.image_signal.connect(self.display_RS_Cam)
                self.RS_Cam_thread.start()
            else:
                self.ui.cameraLabel.setText("No Connected Camera")
                
    def close_RS_Cam(self):
        if self.RS_Cam_thread is not None and self.RS_Cam_thread.isRunning():
            if hasattr(self, "RS_Cam_thread"):
                self.RS_Cam_thread.close()
        if self.RS_Cam_Cali_thread is not None and self.RS_Cam_Cali_thread.isRunning():
            if hasattr(self, "RS_Cam_Cali_thread"):
                self.RS_Cam_Cali_thread.close()
        if self.RS_Cam_battery_thread is not None and self.RS_Cam_battery_thread.isRunning():
            if hasattr(self, "RS_Cam_Cali_thread"):
                self.RS_Cam_battery_thread.close()


    def emergency_RS_Cam(self):
        if self.RS_Cam_thread is not None and self.RS_Cam_thread.isRunning():
            if hasattr(self, "RS_Cam_thread"):
                self.RS_Cam_thread.close()
        if self.RS_Cam_Cali_thread is not None and self.RS_Cam_Cali_thread.isRunning():
            if hasattr(self, "RS_Cam_Cali_thread"):
                self.RS_Cam_Cali_thread.close()
        if self.RS_Cam_battery_thread is not None and self.RS_Cam_battery_thread.isRunning():
            if hasattr(self, "RS_Cam_Cali_thread"):
                self.RS_Cam_battery_thread.close()

    def display_RS_Cam(self, image):
        if self.cameraBtn_status == True:
            if image.all() != None :
                tmpImg = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                qimage = QImage(tmpImg, image.shape[1], image.shape[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                self.ui.cameraLabel.setPixmap(pixmap)
            else:
                self.ui.cameraLabel.setPixmap(QPixmap.fromImage(QImage('')))
                self.ui.cameraLabel.setText("No Connected Camera")
    
    def display_img_on_label(self, img, label):
        self.pixMap, w, h = self.convert_to_pixmap(img)
        # self.pixMap.scaled(w, h, QtCore.Qt.KeepAspectRatio)
        label.setPixmap(self.pixMap)
        label.setAlignment(QtCore.Qt.AlignTop)
        label.setScaledContents(True)
        # label.setFixedWidth(w)
        # label.setFixedHeight(h)
        label.show()
        return
    
    def convert_to_pixmap(self, img):
        tmpImg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        height, width, channel = img.shape
        byteValue = channel * width
        # Convert cv image to pixmap
        tmpQImage =  QtGui.QImage(tmpImg, width, height, byteValue, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(tmpQImage)
        return pixmap, width, height      
                
    ###############   Calib  ###############
    def update_gui_image(self, img):
        self.display_on_label(img, self.ui.cameraLabel)
    
    def getImage(self):
        self.imgPath = data_path+"/BubbleSample/emergency.jpg"
        self.cvImage = cv.resize(cv.imread(self.imgPath), self.InputImgResolution)
        if self.cvImage is None:
            self.ui.cameraLabel.setText("No Image. File Path is not exist.")
            return
        self.cvBubble = cv.resize(self.cvImage, self.BubbleImgResolution)
        return
    
    def Calibrate_RS_Cam_With_Aruco(self, image):      
        if self.calibratedBtn_status == True:
            if image.all() != None :
                tmpImg = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                qimage = QImage(tmpImg, image.shape[1], image.shape[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                self.ui.cameraLabel.setPixmap(pixmap)
            else:
                self.ui.cameraLabel.setPixmap(QPixmap.fromImage(QImage('')))
                self.ui.cameraLabel.setText("No Connected Camera")
    
    def calibrated_coordinate(self):
        if self.RS_Cam_Cali_thread is None  or not self.RS_Cam_Cali_thread.isRunning():
            self.RS_Cam_Cali_thread = QT_RealsenseCaliThread()
        if self.RS_Cam_Cali_thread.check_camera_connection():
            self.RS_Cam_Cali_thread.image_signal.connect(self.Calibrate_RS_Cam_With_Aruco)
            self.RS_Cam_Cali_thread.calib_coord_info_tuple_signal.connect(self.display_calib_coord_info)
            self.RS_Cam_Cali_thread.start()
        else:
            self.ui.cameraLabel.setText("No Connected Camera")
        return 
    
    def display_calib_coord_info(self, coord):
        self.ui.O_point_x_coord_value_label_3.setText("{:.4f}".format(coord[0]))
        self.ui.O_point_y_coord_value_label_3.setText("{:.4f}".format(coord[1]))
        self.ui.X_x_coord_value_label_3.setText("{:.4f}".format(coord[2]))
        self.ui.X_y_coord_value_label_3.setText("{:.4f}".format(coord[3]))
        self.ui.Y_x_coord_value_label_3.setText("{:.4f}".format(coord[4]))
        self.ui.Y_y_coord_value_label_3.setText("{:.4f}".format(coord[5]))
    
    
    def ShowRobotCalibration(self):
        # 將其他按鈕設為不可被按下
        self.ui.cameraBtn.setDisabled(True)
        self.ui.calibratedBtn.setDisabled(True)
        self.ui.bubbleBtn.setDisabled(True)
        self.ui.stopBtn.setDisabled(True)
 
        # 提示視窗
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText("請先透過機械手臂API對Aruco進行手臂校正<br><br>透過手臂教導器移動手臂對到QR code的原點,X點,Y點定出新座標系<br><br>確認手臂校正完畢後，按下確認")
        msg_box.setWindowTitle("請先進行手臂端校正")
        msg_box.setStandardButtons(QMessageBox.Ok)
        
        # 设置消息框字体
        font = QFont("Arial", 13, QFont.Bold)
        msg_box.setFont(font)
        # 设置消息框样式
        msg_box.setStyleSheet("""
            QMessageBox {
                min-width: 400px;
                min-height: 200px;
            }
            QLabel {
                min-width: 350px;
                min-height: 150px;
                padding: 5px;
            }
            QPushButton {
                min-width: 80px;
                min-height: 30px;
            }
        """)
        
        # 如果Ok被按下 就執行鏡頭校正程式
        ret = msg_box.exec()
        if ret == QtWidgets.QMessageBox.Ok:
            self.startCalibration()
            
    def startCalibration(self):
        self.calibrated_coordinate()
        self.calibration_timer.start(10000)  # 10 seconds

    def stopCalibration(self):
        self.close_RS_Cam()   
        self.calibratedBtn_status = False 
        self.calibration_timer.stop()
        
        # 提示視窗
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText("鏡頭校正完畢，可以開始進行夾取")
        msg_box.setWindowTitle("鏡頭校正完畢")
        msg_box.setStandardButtons(QMessageBox.Ok)
        # 设置消息框字体
        font = QFont("Arial", 13, QFont.Bold)
        msg_box.setFont(font)
        # 设置消息框样式
        msg_box.setStyleSheet("""
            QMessageBox {
                min-width: 400px;
                min-height: 200px;
            }
            QLabel {
                min-width: 350px;
                min-height: 150px;
                padding: 5px;
            }
            QPushButton {
                min-width: 80px;
                min-height: 30px;
            }
        """)
        self.ui.cameraBtn.setDisabled(False)
        self.ui.calibratedBtn.setDisabled(False)
        self.ui.bubbleBtn.setDisabled(False)
        self.ui.stopBtn.setDisabled(False)
        
        # 顯示提示視窗
        msg_box.exec_()  
          
    ###############   battery  ###############
     
    def battery_process(self):
        if self.RS_Cam_battery_thread is None  or not self.RS_Cam_battery_thread.isRunning():
            rs_container = RS()
            self.RS_Cam_battery_thread = YOLODetector(rs_cam = rs_container)
        if self.RS_Cam_battery_thread.check_camera_connection():
            self.RS_Cam_battery_thread.update_image_signal.connect(self.RS_Cam_With_battery)
            self.RS_Cam_battery_thread.start()
        else:
            self.ui.cameraLabel.setText("No Connected Camera")
        return 
        # self.yolo_detector.run(update_callback = self.update_gui_image)
        # return
    
    def RS_Cam_With_battery(self, image):      
        if self.batteryBtn_status == True:
            if image.all() != None :
                tmpImg = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                qimage = QImage(tmpImg, image.shape[1], image.shape[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                self.ui.cameraLabel.setPixmap(pixmap)
            else:
                self.ui.cameraLabel.setPixmap(QPixmap.fromImage(QImage('')))
                self.ui.cameraLabel.setText("No Connected Camera")

    ###############   STOP  ###############
    def show_stopImage(self):
        self.imgResolution = self.InputImgResolution
        self.cvImage = cv.imread(data_path+"/BubbleSample/123.png")
        if self.cvImage is None:
            self.ui.cameraLabel.setText("No Image. File Path is not exist.")
            return
        self.cvBubble = cv.resize(self.cvImage, self.imgResolution)
        self.display_img_on_label(self.cvBubble, self.ui.cameraLabel)
        return  

    def display_battery_grasp_info(self):
        df = pd.read_csv(self.csv_file_path)
        # 獲取 Count 列的值
        coord = df['Count'].tolist()
        self.ui.circle_text_print.setText("{:d}".format(coord[0]))
        self.ui.three_L_text_print.setText("{:d}".format(coord[1]))
        self.ui.AL_text_print.setText("{:d}".format(coord[2]))
        self.ui.Li_text_print.setText("{:d}".format(coord[3]))
        self.ui.NICD_text_print.setText("{:d}".format(coord[4]))
        self.ui.NIMH_text_print.setText("{:d}".format(coord[5]))
        self.ui.ZnMn_text_print.setText("{:d}".format(coord[6]))
        self.ui.square_text_print.setText("{:d}".format(coord[7]))
        self.ui.package_text_print.setText("{:d}".format(coord[8]))
        
    ###############   EMERGENCY  ###############
    def show_emergencyImage(self):
        self.imgResolution = self.InputImgResolution
        self.cvImage = cv.imread(data_path+"/BubbleSample/123.png")
        if self.cvImage is None:
            self.ui.cameraLabel.setText("No Image. File Path is not exist.")
            return
        self.cvBubble = cv.resize(self.cvImage, self.imgResolution)
        self.display_img_on_label(self.cvBubble, self.ui.cameraLabel)
        return           

    def ShowEmergencyDialog(self):
        # 將其他按鈕設為不可被按下
        self.ui.cameraBtn.setDisabled(True)
        self.ui.calibratedBtn.setDisabled(True)
        self.ui.bubbleBtn.setDisabled(True)
        self.ui.stopBtn.setDisabled(True)
 
        # 提示視窗
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText("問題是否已排除")
        msg_box.setWindowTitle("緊急狀況")
        msg_box.setStandardButtons(QMessageBox.Yes)
        font = QFont("Arial", 13, QFont.Bold)
        msg_box.setFont(font)
        # 设置消息框样式
        msg_box.setStyleSheet("""
            QMessageBox {
                min-width: 400px;
                min-height: 200px;
            }
            QLabel {
                min-width: 350px;
                min-height: 150px;
                padding: 5px;
            }
            QPushButton {
                min-width: 80px;
                min-height: 30px;
            }
        """)
        # 只有在提示視窗被按下Yes後，才會將其他按鈕的功能開放
        msg_box.buttonClicked.connect(lambda: self.HandleEmergencyDialogButtonClick())
        msg_box.exec_()

    
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = PyQt_MVC_Main()
    
    app.aboutToQuit.connect(lambda: main.close_RS_Cam())

    sys.exit(app.exec_())
    # main.ftpClose()
    
if __name__ == "__main__":
    main()
