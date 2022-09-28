# version 0.0
# Jose Cuaran


import math
import numpy as np
import rclpy
from rclpy.node import Node
#from rclpy.clock import Clock

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from std_msgs.msg import String, Float32
from nav_msgs.msg import Odometry
from mobile_robotics.utils import quaternion_from_euler, lonlat2xyz #edit according to your package's name


class OdometryNode(Node):
    # Initialize some variables
    
    gyro_yaw = 0.0
    gyro_roll = 0.0
    gyro_pitch = 0.0
    blspeed = 0.0 #back left wheel speed
    flspeed = 0.0 #front left wheel speed
    brspeed = 0.0 #back right wheel speed
    frspeed = 0.0 #front right wheel speed
    accel_x = 0.0
    accel_y = 0.0
    accel_z = 0.0
    lat = 0.0
    lon = 0.0
    lat0 = 0.0
    lon0 = 0.0
    flag_initial_lat = False
    flag_initial_lon = False

    x = 0.0 # x robot's position
    y = 0.0 # y robot's position
    theta = 0.0 # heading angle
    l_wheels = 0.3 # Distance between right and left wheels

    last_time = 0.0
    current_time = 0.0

    def __init__(self):
        super().__init__('minimal_subscriber')
        
        # Declare subscribers to all the topics in the rosbag file, like in the example below. Add the corresponding callback functions.
        self.subscription_Gyro_yaw = self.create_subscription(Float32, 'Gyro_yaw', self.callback_Gy, 10)
        # your code here
        self.subscription_Gyro_roll = self.create_subscription(Float32, 'Gyro_roll', self.callback_Gr, 10)
        self.subscription_Gyro_pitch = self.create_subscription(Float32, 'Gyro_pitch', self.callback_Gp, 10)
        self.subscription_Accel_x = self.create_subscription(Float32, 'Accelx', self.callback_Ax, 10)
        self.subscription_Accel_y = self.create_subscription(Float32, 'Accely', self.callback_Ay, 10)
        self.subscription_Accel_z = self.create_subscription(Float32, 'Accelz', self.callback_Az, 10)
        self.subscription_Latitude = self.create_subscription(Float32, 'latitude', self.callback_lat, 10)
        self.subscription_Longitude = self.create_subscription(Float32, 'longitude', self.callback_lon, 10)
        self.subscription_Blspeed = self.create_subscription(Float32, 'Blspeed', self.callback_Blspeed, 10)
        self.subscription_Brspeed = self.create_subscription(Float32, 'Brspeed', self.callback_Brspeed, 10)
        self.subscription_Flspeed = self.create_subscription(Float32, 'Flspeed', self.callback_Flspeed, 10)
        self.subscription_Frspeed = self.create_subscription(Float32, 'Frspeed', self.callback_Frspeed, 10)


        self.last_time = self.get_clock().now().nanoseconds/1e9
        
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10) #keep in mind how to declare publishers for next assignments
        self.timer = self.create_timer(0.1, self.timer_callback_odom) #It creates a timer to periodically publish the odometry.
        
        self.tf_broadcaster = TransformBroadcaster(self) # To broadcast the transformation between coordinate frames.


        self.file_object_results  = open("results_part2.txt", "w+")
        self.timer2 = self.create_timer(0.1, self.callback_write_txt_file) #Another timer to record some results in a .txt file
        

    # Accel and gyro
    def callback_Gy(self, msg):
        self.gyro_yaw = msg.data
    def callback_Gr(self, msg):
        self.gyro_roll = msg.data
    def callback_Gp(self, msg):
        self.gyro_pitch = msg.data
    def callback_Ax(self, msg):
        self.accel_x = msg.data
    def callback_Ay(self, msg):
        self.accel_y = msg.data
    def callback_Az(self, msg):
        self.accel_z = msg.data

    # odometry speed
    def callback_Blspeed(self, msg):
        self.blspeed = msg.data
    def callback_Brspeed(self, msg):
        self.brspeed = msg.data
    def callback_Flspeed(self, msg):
        self.flspeed = msg.data
    def callback_Frspeed(self, msg):
        self.frspeed = msg.data

    # gps
    def callback_lat(self, msg):
        self.lat = msg.data

        #grab initial value
        if not self.flag_initial_lat and self.lat != 0.0:
            self.lat0 = self.lat
            self.flag_initial_lat = True
            print("Initial Lat:", self.lat0)

    def callback_lon(self, msg):
        self.lon = msg.data

        #grab initial value
        if not self.flag_initial_lon and self.lon != 0.0:
            self.lon0 = self.lon
            self.flag_initial_lon = True
            print("Initial Lon:", self.lon0)

    # odometry algorithm
    def timer_callback_odom(self):
        '''
        Compute the linear and angular velocity of the robot using the differential-drive robot kinematics
        Perform Euler integration to find the position x and y of the robot
        '''

        self.current_time = self.get_clock().now().nanoseconds/1e9
        dt = self.current_time - self.last_time # DeltaT
        
        vl = (self.blspeed + self.flspeed)/2.0  #Average Left-wheels speed
        vr = (self.brspeed + self.frspeed)/2.0  # ... Your code here. Average right-wheels speed
        

        v = (vr + vl)/2.0 # ... Linear velocity of the robot
        w = (vr - vl)/(self.l_wheels) # ... Angular velocity of the robot
        self.theta = self.theta + self.gyro_yaw*dt # ...Heading angle...... integrate gyro to get angle
        # self.x = self.x + v*np.cos(self.theta)*dt # ...Position
        # self.y = self.y + v*np.sin(self.theta)*dt # ...Position

        # GPS method
        self.x, self.y = lonlat2xyz(self.lat, self.lon, self.lat0, self.lon0)

        position = [self.x, self.y, 0.0]
        quater = quaternion_from_euler(0.0, 0.0, self.theta) #roll, pitch, yaw

        # print("v, w", v, w)
        # print("theta:",self.theta)
        # print("position: ", position)
        # print("orientation: ", quater)


        # We need to set an odometry message and publish the transformation between two coordinate frames
        # Further info about odometry message: https://docs.ros2.org/foxy/api/nav_msgs/msg/Odometry.html
        # Further info about tf2: https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Introduction-To-Tf2.html
        # Further info about coordinate frames in ROS: https://www.ros.org/reps/rep-0105.html

        frame_id = 'odom'
        child_frame_id = 'base_link'
        
        
        self.broadcast_tf(position, quater, frame_id, child_frame_id)  # Before creating the odometry message, go to the broadcast_tf function and complete it.
        
        odom = Odometry()
        odom.header.frame_id = frame_id
        odom.header.stamp = self.get_clock().now().to_msg()

        # set the pose. Uncomment next lines

        odom.pose.pose.position.x = position[0] # ...
        odom.pose.pose.position.y = position[1] # ...
        odom.pose.pose.position.z = 0.0 # ... 
        odom.pose.pose.orientation.x = quater[0]
        odom.pose.pose.orientation.y = quater[1] # ...
        odom.pose.pose.orientation.z = quater[2] # ...
        odom.pose.pose.orientation.w = quater[3] # ...

        # set the velocities. Uncomment next lines
        odom.child_frame_id = child_frame_id
        odom.twist.twist.linear.x = v*np.cos(self.theta) # ...
        odom.twist.twist.linear.y = v*np.sin(self.theta) # ...
        odom.twist.twist.linear.z = 0.0 # ...
        odom.twist.twist.angular.x =  self.gyro_roll # ...
        odom.twist.twist.angular.y = self.gyro_yaw # ...
        odom.twist.twist.angular.z = self.gyro_pitch # ...

        self.odom_pub.publish(odom)

        self.last_time = self.current_time
        
    def broadcast_tf(self, pos, quater, frame_id, child_frame_id):
        '''
        It continuously publishes the transformation between two reference frames.
        Complete the translation and the rotation of this transformation
        '''
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame_id

        # Uncomment next lines and complete the code
        t.transform.translation.x = pos[0] # ...
        t.transform.translation.y = pos[1] # ...
        t.transform.translation.z = pos[2] # ...

        t.transform.rotation.x = quater[0] # ...
        t.transform.rotation.y = quater[1] # ...
        t.transform.rotation.z = quater[2] # ...
        t.transform.rotation.w = quater[3] # ...

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)
    
    def callback_write_txt_file(self):
        if (self.x != 0 or self.y != 0 or self.theta != 0):
            self.file_object_results.write(str(self.current_time) + " " + str(self.x)+" "+str(self.y)+" "+str(self.theta)+"\n")

    
def main(args=None):
    rclpy.init(args=args)

    odom_node = OdometryNode()

    rclpy.spin(odom_node)
    odom_node.file_object_results.close()
    odom_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
