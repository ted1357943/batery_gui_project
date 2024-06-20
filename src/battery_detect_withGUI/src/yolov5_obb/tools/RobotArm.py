import re
import rospy
import sys
# sys.path.append('/home/yang/jojo/battery_project/ros_test/src/tmr_ros1')
from tm_msgs.msg import *
from tm_msgs.srv import *
import argparse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time as time

class RobotController:
    def __init__(self):
        rospy.init_node('robot_controller')
        rospy.Subscriber('tm_driver/svr_response', SvrResponse, self.callback)
        rospy.wait_for_service('/tm_driver/send_script')
        rospy.wait_for_service('/tm_driver/set_io')
        rospy.wait_for_service('tm_driver/ask_item')
        self.send_script_service = rospy.ServiceProxy('/tm_driver/send_script', SendScript)
        self.set_io_service = rospy.ServiceProxy('/tm_driver/set_io', SetIO)
        self.ask_item_service = rospy.ServiceProxy('tm_driver/ask_item', AskItem)
    
    def send_script(self, script):
        try:
            move_cmd = SendScriptRequest()
            move_cmd.script = script
            resp = self.send_script_service(move_cmd)
            return resp
        except rospy.ServiceException as e:
            print("Send script service call failed: %s" % e)

    def wait(self, seconds=3.0):
        self.send_script("QueueTag(1,1)")
        while True:
            resp = self.send_script("QueueTag(1,1)")
            if resp:
                break
        rospy.sleep(seconds)

    def set_io(self, state):
        try:
            io_cmd = SetIORequest()
            io_cmd.module = 1
            io_cmd.type = 1
            io_cmd.pin = 0
            io_cmd.state = state
            resp = self.set_io_service(io_cmd)
            return resp
        except rospy.ServiceException as e:
            print("IO service call failed: %s" % e)

    def callback(self, data):
        rospy.loginfo(rospy.get_caller_id() + ': id: %s, content: %s\n', data.id, data.content)

    def ask_item_demo(self):
        res = self.ask_item_service('he0', 'Coord_Robot_Tool', 1)
        position = res.value
        a,b = position.split("=", 1)
        b = b.strip('{}')
        b = b.split(',', 5)
        c = [round(float(i)) for i in b]
        print("pos:", c)
        return c

    def ask_base_value(self):
        res = self.ask_item_service('he0', 'Base_Value', 1)
        position = res.value
        a,b = position.split("=", 1)
        b = b.strip('{}')
        b = b.split(',', 5)
        c = [round(float(i)) for i in b]
        print("pos:", c)
        return c
    
    def grasp_and_move(self, classnumber, center_tfx, center_tfy, angle_str,ChangeBase=False):
        if ChangeBase:
            self.send_script("ChangeBase(\"123\")")
            self.ask_base_value()
        if (classnumber[-1] == 1):
            self.set_io(0.0)
            #move to the center of the cylinder
            self.send_script(f"PTP(\"CPP\",{center_tfx}, {center_tfy},-20, 0, 0,{angle_str},100,10,0,false)")
            #move to the top of the cylinder
            self.send_script(f"PTP(\"CPP\",{center_tfx}, {center_tfy},2 , 0 , 0 ,{angle_str},100,10,0,false)")
            time.sleep(5)
            self.set_io(1.0)
            time.sleep(8)
            #put the cylinder in the box
            self.send_script("PTP(\"CPP\",-809.22,-123.72,-260.42,-1.49,-3.04,-1.7,100,10,0,false)")
            time.sleep(5)
            self.set_io(0.0)
            time.sleep(3)
            #initial position
            self.go_home()

        elif (classnumber[-1] == 2):    
            self.set_io(0.0)
            self.send_script(f"PTP(\"CPP\",{center_tfx}, {center_tfy},-20, 0, 0,{angle_str},100,10,0,false)")
            self.send_script(f"PTP(\"CPP\",{center_tfx}, {center_tfy},2 , 0 , 0 ,{angle_str},100,10,0,false)")
            time.sleep(5)
            self.set_io(1.0)
            time.sleep(8)
            #put the package in the box
            self.send_script("PTP(\"CPP\",-809.22,-123.72,-260.42,-1.49,-3.04,-1.7,100,10,0,false)")
            time.sleep(5)
            self.set_io(0.0)
            time.sleep(3)            
            #initial position
            self.go_home()          	
                            
        elif (classnumber[-1] == 3):    
            self.set_io(0.0)
            self.send_script(f"PTP(\"CPP\",{center_tfx}, {center_tfy},-20, 0, 0,{angle_str},100,10,0,false)")
            self.send_script(f"PTP(\"CPP\",{center_tfx}, {center_tfy},2 , 0 , 0 ,{angle_str},100,10,0,false)")
            time.sleep(5)
            self.set_io(1.0)
            time.sleep(8)
            #put the square in the box
            self.send_script("PTP(\"CPP\",-809.22,-123.72,-260.42,-1.49,-3.04,-1.7,100,10,0,false)")
            time.sleep(5)
            self.set_io(0.0)
            time.sleep(3)            
            #initial position
            self.go_home()
               
    def emergency_stop(self):
        self.send_script("Exit(false)")
        
    def go_home(self):
        self.send_script("PTP(\"CPP\",-361.79, -32.53, -272.46, -4.99, -3.02, -20,100,10,0,false)") 