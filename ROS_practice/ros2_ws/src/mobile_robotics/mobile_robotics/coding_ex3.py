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

class CodingExercise3(Node):
   

    
    def __init__(self):
        super().__init__('CodingExercise3')

        self.ranges = [] # lidar measurements
        
        self.point_list = [] # A list of points to draw lines
        self.line = Marker()
        self.line_marker_init(self.line)


        # Ros subscribers and publishers
        self.subscription_ekf = self.create_subscription(Odometry, 'terrasentia/ekf', self.callback_ekf, 10)
        self.subscription_scan = self.create_subscription(LaserScan, 'terrasentia/scan', self.callback_scan, 10)
        self.pub_lines = self.create_publisher(Marker, 'lines', 10)
        self.timer_detect_line_example = self.create_timer(0.01, self.detect_line_callback)
        self.timer_draw_line = self.create_timer(0.1, self.draw_line_callback)
        self.point_set = []
        self.position = None
    def callback_ekf(self, msg):
        # You will need this function to read the translation and rotation of the robot with respect to the odometry frame
        self.position = msg.pose.pose.position
        self.orient = msg.pose.pose.orientation
   
    def callback_scan(self, msg):
        if(self.position == None):
            return
        self.ranges = list(msg.ranges) # Lidar measurements
        n = len(self.ranges)
        self.point_set = np.zeros((n, 3))
        r = R.from_quat([self.orient.x, self.orient.y, self.orient.z, self.orient.w])
        r2 = R.from_euler('z', -90, degrees=True)
        for i in range(n):
            self.point_set[i][0] = self.ranges[i] * math.cos(-math.pi/4 + math.pi*3/2/(n-1)*i)
            self.point_set[i][1] = self.ranges[i] * math.sin(-math.pi/4 + math.pi*3/2/(n-1)*i)
            self.point_set[i][2] = 0
        self.point_set = r.apply(self.point_set)
        self.point_set = r2.apply(self.point_set)
        self.point_set += np.array([self.position.x, self.position.y, self.position.z])
        np.save('./laser', self.point_set)
        # print("some-ranges:", self.ranges[0:5])
        # print("Number of ranges:", len(self.ranges))
        # print("some points: \n", self.point_set[0:5])
    def draw_line_callback(self):
        self.pub_lines.publish(self.line)
        pass
    def append_line(self, x1, y1, z1, x2, y2, z2):
        p0 = Point()
        p0.x = x1
        p0.y = y1
        p0.z = z1

        p1 = Point()
        p1.x = x2
        p1.y = y2
        p1.z = z2

        # self.point_list = []
        # self.point_list.append(copy(p0)) 
        # self.point_list.append(copy(p1)) # You can append more pairs of points
        # self.line.points = self.point_list

        self.line.points.append(copy(p0)) 
        self.line.points.append(copy(p1)) # You can append more pairs of points

        # self.pub_lines.publish(self.line) # It will draw a line between each pair of points
    def detect_line_callback(self):
        # Here is just a simple example on how to draw a line on rviz using line markers. Feel free to use any other method
        
        if(len(self.point_set) == 0):
            return
        t1 = time.time()
        point_set = np.array(self.point_set)
        n = point_set.shape[0]
        line_set = []
        def split(point_set, start, end, threshold):
            point_start = point_set[start]
            point_end = point_set[end]
            line = np.cross(point_start, point_end)
            if(end - start < 5):
                # append_line(point_start[0], point_start[1], point_start[2], point_end[0], point_end[1], point_start[2])
                return
            q = np.sqrt(np.sum(line[0:2]**2))
            max_d = -float('inf')
            max_ind = -1
            for i in range(start+1, end):
                dist = np.abs(np.sum(point_set[i] * line)) / q
                if(dist >= max_d):
                    max_d = dist
                    max_ind = i
            if(max_d > threshold):
                split(point_set, start, max_ind-1, threshold)
                split(point_set, max_ind+1, end, threshold)
            else:
                robot_pos = np.array([self.position.x, self.position.y, self.position.z])
                np.save("./robot_pos", robot_pos)
                start_d = np.sqrt(np.sum((point_start - robot_pos))**2)
                end_d = np.sqrt(np.sum((point_end - robot_pos)**2))
                if(start_d > 20 or end_d > 20):
                    pass
                else:
                    self.append_line(point_start[0], point_start[1], point_start[2], point_end[0], point_end[1], point_start[2])

        split(point_set, 100, n-1, 0.08)
        t2 = time.time()
        print("Time to estimate line: ", t2-t1)
        # line_set = np.array(line_set)

    def line_marker_init(self, line):
        line.header.frame_id="/odom"
        line.header.stamp=self.get_clock().now().to_msg()

        line.ns = "markers"
        line.id = 0

        line.type=Marker.LINE_LIST
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0

        line.scale.x = 0.05
        line.scale.y= 0.05
        
        line.color.r = 1.0
        line.color.a = 1.0
        #line.lifetime = 0


def main(args=None):
    rclpy.init(args=args)

    cod3_node = CodingExercise3()
    
    rclpy.spin(cod3_node)

    cod3_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
