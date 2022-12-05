# Student name: 

import math
import numpy as np
import rclpy
from rclpy.node import Node
import time
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, PoseStamped, TransformStamped
from std_msgs.msg import String, Float32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
import matplotlib.pyplot as plt
import time
from tf2_msgs.msg import TFMessage
from copy import copy
from visualization_msgs.msg import Marker

# Further info:
# On markers: http://wiki.ros.org/rviz/DisplayTypes/Marker
# Laser Scan message: http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/LaserScan.html

class Final(Node):
   

    
    def __init__(self):
        super().__init__('Final')

        # Ros subscribers and publishers
        self.subscription_ekf = self.create_subscription(Odometry, 'terrasentia/ekf', self.callback_ekf, 10)
        self.sub_rtan_odom = self.create_subscription(Odometry, 'rtabmap/odom', self.callback_rtab, 10)

        self.ekf_data = np.zeros((0, 3))
        self.rtab_data = np.zeros((0, 3))
    def callback_rtab(self, msg):
        self.position_rtab = msg.pose.pose.position
        self.orient_rtab = msg.pose.pose.orientation
        self.rtab_data = np.concatenate((self.rtab_data, np.array([[self.position_rtab.x, self.position_rtab.y, self.position_rtab.z]])), axis=0)
        np.savetxt("rtab.txt", self.rtab_data)
    def callback_ekf(self, msg):
        # You will need this function to read the translation and rotation of the robot with respect to the odometry frame
        self.position_ekf = msg.pose.pose.position
        self.orient_ekf = msg.pose.pose.orientation
        self.ekf_data = np.concatenate((self.ekf_data, np.array([[self.position_ekf.x, self.position_ekf.y, self.position_ekf.z]])), axis=0)
        # np.savetxt("ekf.txt", self.ekf_data)
# ros2 run tf2_ros static_transform_publisher 0.0 0.0 0.0 -1.5708 0.0 -1.5708 zed2_imu_link zed2_left_camera_optical_frame
# ros2 launch rtabmap_ros rtabmap.launch.py tabmap_args:="--delete_db_on_start" rgb_topic:=/z/left/color depth_topic:=/terrasentia/zed2/zed_node/depth/depth_registered camera_info_topic:=/terrasentia/zed2/zed_node/left/camera_info frame_id:=zed2_imu_link approx_sync:=false wait_imu_to_init:=true imu_topic:=/terrasentia/zed2/zed_node/imu/data qos:=1
def main(args=None):
    rclpy.init(args=args)

    cod3_node = Final()
    
    rclpy.spin(cod3_node)

    cod3_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
