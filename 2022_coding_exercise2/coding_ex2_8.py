# Student name: 

import math
from turtle import position
import numpy as np
from numpy import linalg as LA
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Accel
from tf2_ros import TransformBroadcaster

from std_msgs.msg import String, Float32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import matplotlib.pyplot as plt
import time
from mobile_robotics.utils import quaternion_from_euler, lonlat2xyz, quat2euler


class ExtendedKalmanFilter(Node):

    
    def __init__(self):
        super().__init__('ExtendedKalmanFilter')
        self.x = self.y = self.xPrev = self.yPrev = self.gps_heading = 0.0
        
        #array to save the sensor measurements from the rosbag file
        #measure = [p, q, r, fx, fy, fz, x, y, z, vx, vy, vz] 
        self.measure = np.zeros(12)
        
        #Initialization of the variables used to generate the plots.
        self.PHI = []  
        self.PSI = []
        self.THETA = []
        self.P_R = []
        self.P_R1 = []
        self.P_R2 = []
        self.Pos = []
        self.Vel = []
        self.Quater = []
        self.measure_PosX = []
        self.measure_PosY = []
        self.measure_PosZ = []
        self.P_angular = []
        self.Q_angular = []
        self.R_angular = []
        self.P_raw_angular = []
        self.Q_raw_angular = []
        self.R_raw_angular = []
        self.Bias =[]
        
        self.POS_X = []
        self.POS_Y = []
        
        
        #Initialization of the variables used in the EKF
        
        # initial bias values, these are gyroscope and accelerometer biases
        self.bp= 0.0
        self.bq= 0.0
        self.br= 0.0
        self.bfx = 0.0
        self.bfy = 0.0
        self.bfz = 0.0
        # initial rotation
        self.q2, self.q3, self.q4, self.q1 = quaternion_from_euler(0.0, 0.0, np.pi/2) #[qx,qy,qz,qw]

        #initialize the state vector [x y z vx vy vz          quat    gyro-bias accl-bias]
        self.xhat = np.array([[0, 0, 0, 0, 0, 0, self.q1, self.q2, self.q3, self.q4, self.bp, self.bq, self.br, self.bfx, self.bfy, self.bfz]]).T

        self.rgps=np.array([-0.15, 0 ,0]) #This is the location of the GPS wrt CG, this is very important
        
        #noise params process noise (my gift to you :))
        Q_trust = 1
        self.Q = Q_trust * np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.5, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
        #measurement noise
        #GPS position and velocity
        R_trust = 1
        self.R = R_trust * np.diag([10, 10, 10, 2, 2, 2])
        
       
        #Initialize P, the covariance matrix
        self.P = np.diag([30, 30, 30, 3, 3, 3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.Pdot=self.P*0.0
        
        self.time = []
        self.loop_t = 0

        # You might find these blocks useful when assembling the transition matrices
        self.Z = np.zeros((3,3))
        self.I = np.eye(3,3)
        self.Z34 = np.zeros((3,4))
        self.Z43 = np.zeros((4,3))
        self.Z36 = np.zeros((3,6))

        self.lat = 0
        self.lon = 0
        self.lat0 = 0
        self.lon0 = 0
        self.flag_lat = False
        self.flag_lon = False
        self.dt = 0.0125 #set sample time

        # Ros subscribers and publishers
        self.subscription_imu = self.create_subscription(Imu, 'terrasentia/imu', self.callback_imu, 10)
        self.subscription_gps_lat = self.create_subscription(Float32, 'gps_latitude', self.callback_gps_lat, 10)
        self.subscription_gps_lon = self.create_subscription(Float32, 'gps_longitude', self.callback_gps_lon, 10)
        self.subscription_gps_speed_north = self.create_subscription(Float32, 'gps_speed_north', self.callback_gps_speed_north, 10)
        self.subscription_gps_speend_east = self.create_subscription(Float32, 'gps_speed_east', self.callback_gps_speed_east, 10)
        
        # Question 5: create publisher for odometry...
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10) #keep in mind how to declare publishers for next assignments
        self.timer = self.create_timer(0.1, self.timer_callback_odom) #It creates a timer to periodically publish the odometry.
        self.tf_broadcaster = TransformBroadcaster(self) # To broadcast the transformation between coordinate frames.

        self.timer_ekf = self.create_timer(self.dt, self.ekf_callback)
        self.timer_plot = self.create_timer(2, self.plot_data_callback)

    
    def callback_imu(self,msg):
        #measurement vector = [p, q, r, fx, fy, fz, x, y, z, vx, vy, vz]
        # In practice, the IMU measurements should be filtered. In this coding exercise, we are just going to clip
        # the values of velocity and acceleration to keep them in physically possible intervals.
        self.measure[0] = np.clip(msg.angular_velocity.x,-5,5) #(-5,5)
        self.measure[1] = np.clip(msg.angular_velocity.y,-5,5) #None #..(-5,5)
        self.measure[2] = np.clip(msg.angular_velocity.z,-5,5) #None #..(-5,5)
        self.measure[3] = np.clip(msg.linear_acceleration.x,-6,6) #None #..(-6,6)
        self.measure[4] = np.clip(msg.linear_acceleration.y,-6,6) #None #..(-6,6)
        self.measure[5] = np.clip(msg.linear_acceleration.z,-16,-4) #None #..(-16,-4) 
 
    def callback_gps_lat(self, msg):
        self.lat = msg.data
        if (self.flag_lat == False): #just a trick to recover the initial value of latitude
            self.lat0 = msg.data
            self.flag_lat = True
        
        if (self.flag_lat and self.flag_lon): 
            x, y = lonlat2xyz(self.lat, self.lon, self.lat0, self.lon0) # convert latitude and longitude to x and y coordinates
            self.measure[6] = x
            self.measure[7] = y
            self.measure[8] = 0.0 

    
    def callback_gps_lon(self, msg):
        self.lon = msg.data
        if (self.flag_lon == False): #just a trick to recover the initial value of longitude
            self.lon0 = msg.data
            self.flag_lon = True    
    
    def callback_gps_speed_east(self, msg): 
        self.measure[9] = msg.data #None # ..
        self.measure[11] = 0.0 # vz

    def callback_gps_speed_north(self, msg):
        self.measure[10] = msg.data #None # vy
        # print("speed north:", self.measure[10])

   
    def ekf_callback(self):
        #print("iteration:  ",self.loop_t)
        if (self.flag_lat and self.flag_lon):  #Trick  to sincronize rosbag with EKF
            self.ekf_function()
        else:
            print("Play the rosbag file...")

    # odometry algorithm
    def timer_callback_odom(self):
        '''
        Compute the linear and angular velocity of the robot using the differential-drive robot kinematics
        Perform Euler integration to find the position x and y of the robot
        '''

        self.current_time = self.get_clock().now().nanoseconds/1e9
        dt = self.dt #self.current_time - self.last_time # DeltaT

        # We need to set an odometry message and publish the transformation between two coordinate frames
        # Further info about odometry message: https://docs.ros2.org/foxy/api/nav_msgs/msg/Odometry.html
        # Further info about tf2: https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Introduction-To-Tf2.html
        # Further info about coordinate frames in ROS: https://www.ros.org/reps/rep-0105.html

        if len(self.POS_X) == 0:
            position = [0.0, 0.0, 0.0]
        else:
            position = [self.POS_X[-1], self.POS_Y[-1], 0.0]

        if len(self.Quater) == 0:
            quater = [0.0, 0.0, 0.0, 0.0]
        else:
            # INS method
            # quater = self.Quater[-1]
            # quater = [self.xhat[4,0], self.xhat[5,0], self.xhat[6,0], self.xhat[3,0]]
            quater = [self.quat[1,0], self.quat[2,0], self.quat[3,0], self.quat[0,0]]
            # quater = [self.quat[0,0], self.quat[1,0], self.quat[2,0], self.quat[3,0]]
            # quater = [self.xhat[4,0], self.xhat[5,0], self.xhat[6,0], self.xhat[3,0]]
            # quater = [self.xhat[3,0], self.xhat[4,0], self.xhat[5,0], self.xhat[6,0]]
            quater = quater/np.linalg.norm(quater) #ensure normalized

            # GPS method
            # self.x, self.y = lonlat2xyz(self.lat, self.lon, self.lat0, self.lon0)

            # if self.x != self.xPrev or self.y != self.yPrev:
            #     diff_x = self.x - self.xPrev
            #     diff_y = self.y - self.yPrev
            #     self.gps_heading = np.arctan2(diff_y,diff_x) + 2*np.pi
            # self.xPrev = self.x
            # self.yPrev = self.y
            # quater = quaternion_from_euler(0.0, 0.0, self.gps_heading)
        # position = [self.xhat[0,0], self.xhat[1,0], self.xhat[2,0]]


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
        odom.twist.twist.linear.x = 0.0 #*np.cos(self.theta) # ...
        odom.twist.twist.linear.y = 0.0 # b/c none-holonomic  v*np.sin(self.theta) # ...
        odom.twist.twist.linear.z = 0.0 # ...
        odom.twist.twist.angular.x = 0.0 #self.gyro_roll # ...
        odom.twist.twist.angular.y = 0.0  #self.gyro_pitch # ...
        odom.twist.twist.angular.z = 0.0  #self.gyro_yaw # ... can also use wheel differential turn

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
        t.transform.translation.x = float(pos[0]) # ...
        t.transform.translation.y = float(pos[1]) # ...
        t.transform.translation.z = float(pos[2]) # ...

        t.transform.rotation.x = float(quater[0]) # ...
        t.transform.rotation.y = float(quater[1]) # ...
        t.transform.rotation.z = float(quater[2]) # ...
        t.transform.rotation.w = float(quater[3]) # ...

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)
    
    def ekf_function(self):
        
        # Adjusting angular velocities and acceleration with the corresponding bias
        self.p = (self.measure[0]-self.xhat[10,0])
        self.q = (self.measure[1]-self.xhat[11,0])
        self.r = self.measure[2]-self.xhat[12,0]
        self.fx = (self.measure[3]-self.xhat[13,0])
        self.fy = (self.measure[4]-self.xhat[14,0])
        self.fz = -self.measure[5]-self.xhat[15,0]
        
        # Get the current quaternion values from the state vector
        # Remember again the state vector [x y z vx vy vz q1 q2 q3 q4 bp bq br bx by bz]
        self.quat = np.array([[self.xhat[6,0], self.xhat[7,0], self.xhat[8,0], self.xhat[9,0]]]).T
    
        self.q1 = self.quat[0,0] #w
        self.q2 = self.quat[1,0] #x
        self.q3 = self.quat[2,0] #y
        self.q4 = self.quat[3,0] #z
                
        # Rotation matrix: body to inertial frame
        self.R_bi = np.array([[pow(self.q1,2)+pow(self.q2,2)-pow(self.q3,2)-pow(self.q4,2), 2*(self.q2*self.q3-self.q1*self.q4), 2*(self.q2*self.q4+self.q1*self.q3)],
                          [2*(self.q2*self.q3+self.q1*self.q4), pow(self.q1,2)-pow(self.q2,2)+pow(self.q3,2)-pow(self.q4,2), 2*(self.q3*self.q4-self.q1*self.q2)],
                          [2*(self.q2*self.q4-self.q1*self.q3), 2*(self.q3*self.q4+self.q1*self.q2), pow(self.q1,2)-pow(self.q2,2)-pow(self.q3,2)+pow(self.q4,2)]])
        
            
        #Prediction step
        #First write out all the dots for all the states, e.g. pxdot, pydot, q1dot etc
       
        # .. your code here
        
        # (eq. 101)
        pxdot = self.xhat[3,0] #basically just vx...?
        pydot = self.xhat[4,0] #
        pzdot = self.xhat[5,0] #
        # (eq. 102)
        # while(1):
            # print(self.R_bi.shape)
        # print(np.dot(self.R_bi, np.array([[self.fx, self.fy, self.fz]]).T))
        [vxdot, vydot, vzdot] = np.dot(self.R_bi, np.array([[self.fx, self.fy, self.fz]]).T)
        # print(vxdot, vydot, vzdot)

        # (eq. 103)
        om = np.array([[0, self.p, self.q, self.r],
                    [-self.p, 0, -self.r, self.q],
                    [-self.q, self.r, 0, -self.p],
                    [-self.r, -self.q, self.p, 0]]) #(eq. 107)
        [q1dot, q2dot, q3dot, q4dot] = -0.5*np.matmul(om, [self.q1, self.q2, self.q3, self.q4]) #(eq. 103)
        bpdot = bqdot = brdot = bxdot = bydot = bzdot = 0 #(eq. 104 & 105)
        
        #Now integrate Euler Integration for Process Updates and Covariance Updates
        # Euler works fine
        # Remember again the state vector [x y z vx vy vz q1 q2 q3 q4 bp bq br bx by bz]
        self.xhat[0,0] = self.xhat[0,0] + self.dt*pxdot   
        self.xhat[1,0] = self.xhat[1,0] + self.dt*pydot #None # ..
        self.xhat[2,0] = self.xhat[2,0] + self.dt*pzdot #None # ..
        self.xhat[3,0] = self.xhat[3,0] + self.dt*vxdot #None # ..
        self.xhat[4,0] = self.xhat[4,0] + self.dt*vydot #None # ..
        self.xhat[5,0] = self.xhat[5,0] + self.dt*(vzdot - 9.801) #None # .. Do not forget Gravity (9.801 m/s2) 
        self.xhat[6,0] = self.xhat[6,0] + self.dt*q1dot #None # ..
        self.xhat[7,0] = self.xhat[7,0] + self.dt*q2dot #None # ..
        self.xhat[8,0] = self.xhat[8,0] + self.dt*q3dot #None # ..
        self.xhat[9,0] = self.xhat[9,0] + self.dt*q4dot #None # ..

        print("x ekf: ", self.xhat[0,0])
        print("y ekf: ", self.xhat[1,0])
        print("z ekf: ", self.xhat[2,0])
        
        # Extract and normalize the quat    
        self.quat = np.array([[self.xhat[6,0], self.xhat[7,0], self.xhat[8,0], self.xhat[9,0]]]).T
        # .. Normailize quat
        self.quat = self.quat/np.linalg.norm(self.quat) #None .. # code here. Uncomment this line

        #re-assign quat
        # print(self.quat[0,0], type(self.quat[0,0]))
        self.xhat[6,0] = self.quat[0,0]
        self.xhat[7,0] = self.quat[1,0]
        self.xhat[8,0] = self.quat[2,0]
        self.xhat[9,0] = self.quat[3,0]

        # Now write out all the partials to compute the transition matrix Phi
        #delV/delQ
        #(eq. 118)
        Fvq = 2*np.array([[(self.q1*self.fx - self.q4*self.fy + self.q3*self.fz), (self.q2*self.fx + self.q3*self.fy + self.q4*self.fz), (-self.q3*self.fx + self.q2*self.fy + self.q1*self.fz) , (-self.q4*self.fx - self.q1*self.fy + self.q2*self.fz)],
                            [(self.q4*self.fx + self.q1*self.fy - self.q2*self.fz),(self.q3*self.fx - self.q2*self.fy - self.q1*self.fz), (self.q2*self.fx + self.q3*self.fy + self.q4*self.fz) , (self.q1*self.fx - self.q4*self.fy + self.q3*self.fz)],
                            [(-self.q3*self.fx + self.q2*self.fy + self.q1*self.fz),(self.q4*self.fx + self.q1*self.fy - self.q2*self.fz),(-self.q1*self.fx + self.q4*self.fy - self.q3*self.fz), (self.q2*self.fx + self.q3*self.fy + self.q4*self.fz)]]) #None # ..
        #delV/del_abias
        #(eq. 114)
        Fvb = -self.R_bi #None # ..
        
        #delQ/delQ
        #(eq. 115)
        Fqq = -0.5*om #None # ..
     
        #delQ/del_gyrobias
        #(eq. 116, 119, and 120)
        Fqb = 0.5*np.array([[self.q2, self.q3, self.q4],
                            [-self.q1, self.q4, -self.q3],
                            [-self.q4, -self.q1, self.q2],
                            [self.q3, -self.q2, -self.q1]])#None # ..
        # Now assemble the Transition matrix A
        zeros_33 = self.Z #np.zeros((3,3))
        zeros_34 = self.Z34 #np.zeros((3,4))
        zeros_43 = self.Z43 #np.zeros((4,3))
        Ident = self.I #np.identity(3)

        temp_row1 = np.concatenate((zeros_33, Ident, zeros_34, zeros_33, zeros_33), axis=1)
        temp_row2 = np.concatenate((zeros_33, zeros_33, Fvq, zeros_33, Fvb), axis = 1)
        temp_row3 = np.concatenate((zeros_43, zeros_43, Fqq, Fqb, zeros_43), axis = 1)
        temp_row4 = np.concatenate((zeros_33, zeros_33, zeros_34, zeros_33, zeros_33), axis = 1)
        temp_row5 = np.concatenate((zeros_33, zeros_33, zeros_34, zeros_33, zeros_33), axis = 1)
        A = np.concatenate((temp_row1, temp_row2, temp_row3, temp_row4, temp_row5), axis = 0)
        # A = np.array([[zeros_33, Ident, zeros_34, zeros_33, zeros_33],
        #                 [zeros_33, zeros_33, Fvq, zeros_33, Fvb],
        #                 [zeros_43, zeros_43, Fqq, Fqb, zeros_43],
        #                 [zeros_33, zeros_33, zeros_34, zeros_33, zeros_33],
        #                 [zeros_33, zeros_33, zeros_34, zeros_33, zeros_33]]) #None # ..


        #Propagate the error covariance matrix, I suggest using the continuous integration since Q, R are not discretized 
        #Pdot = A@P+P@A.transpose() + Q
        #P = P +Pdot*dt
        # print(A.shape, self.P.shape, self.Q.shape)
        self.Pdot = A@self.P+self.P@A.T + self.Q # .. ??????
        self.P = self.P +self.Pdot*self.dt  # ..
        
        #Correction step
        #Get measurements 3 positions and 3 velocities from GPS
        self.z = np.array([[self.measure[6], self.measure[7], self.measure[8], self.measure[9], self.measure[10], self.measure[11]]]).T #x y z vx vy vz
        # self.z = np.ravel(self.z)
    
        #Write out the measurement matrix linearization to get H
        r1 = self.rgps[0]
        
        # del v/del q
        # (eq. 125)
        Hvq = 2*r1*np.array([[(self.q3*self.q + self.q4*self.r), (self.q4*self.q - self.q3*self.r), (self.q1*self.q - self.q2*self.r), (self.q2*self.q + self.q1*self.r)],
                        [(-self.q2*self.q - self.q1*self.r), (self.q2*self.r - self.q1*self.q), (self.q4*self.q - self.q3*self.r), (self.q3*self.q + self.q4*self.r)],
                        [(self.q1*self.q - self.q2*self.r), (-self.q2*self.q - self.q1*self.r), (-self.q3*self.q - self.q4*self.r), (self.q4*self.q - self.q3*self.r)]])#None # ..
        
        #del P/del q
        # (eq. 124)
        Hxq = 2*r1*np.array([[-self.q1, -self.q2, self.q3, self.q4],
                            [-self.q4, -self.q3, -self.q2, -self.q1],
                            [self.q3, -self.q4, self.q1, -self.q2]]) #None # ..
        
        # Assemble H
        # (eq. 126)
        zeros_36 = self.Z36 #np.zeros((3,6))
        temp_row1 = np.concatenate((Ident, zeros_33, Hxq, zeros_36), axis = 1)
        temp_row2 = np.concatenate((zeros_33, Ident, Hvq, zeros_36), axis = 1)
        H = np.concatenate((temp_row1, temp_row2), axis = 0)
        # H = np.array([[Ident, zeros_33, Hxq, zeros_36],
        #                 [zeros_33, Ident, Hvq, zeros_36]]) #None # ..

        #Compute Kalman gain
        # (eq. 65)
        L = self.P@H.T@np.linalg.inv(H@self.P@H.T + self.R) #None # Kalman gain
        
        #Perform xhat correction    xhat = xhat + L@(z-H@xhat)
        self.xhat = self.xhat + L@(self.z - H@self.xhat) #None # .. uncomment
        
        #propagate error covariance approximation P = (np.eye(16,16)-L@H)@P
        self.P = (np.eye(16, 16) - L@H)@self.P #None # ..

        #Now let us do some book-keeping 
        # Get some Euler angles
        
        phi, theta, psi = quat2euler(self.quat.T)

        self.PHI.append(phi*180/math.pi)
        self.THETA.append(theta*180/math.pi)
        self.PSI.append(psi*180/math.pi)
    
          
        # Saving data for the plots. Uncomment the 4 lines below once you have finished the ekf function

        DP = np.diag(self.P)
        self.P_R.append(DP[0:3])
        self.P_R1.append(DP[3:6])
        self.P_R2.append(DP[6:10])
        self.Pos.append(self.xhat[0:3].T[0])
        self.POS_X.append(self.xhat[0,0])
        self.POS_Y.append(self.xhat[1,0])
        self.Vel.append(self.xhat[3:6].T[0])
        self.Quater.append(self.xhat[6:10].T[0])
        self.Bias.append(self.xhat[10:16].T[0])
        B = self.measure[6:9].T
        self.measure_PosX.append(B[0])
        self.measure_PosY.append(B[1])
        self.measure_PosZ.append(B[2])

        self.P_angular.append(self.p)
        self.Q_angular.append(self.q)
        self.R_angular.append(self.r)

        self.loop_t += 1
        self.time.append(self.loop_t*self.dt)
        # print("timer", self.loop_t)

        # Act as pausing to see all the graphs without closing them
        if self.loop_t >= 18000:
            while(1):
                pass

    def plot_data_callback(self):

        plt.figure(1)
        plt.clf()
        plt.plot(self.time,self.PHI,'b', self.time, self.THETA, 'g', self.time,self.PSI, 'r')
        plt.legend(['phi','theta','psi'])
        plt.title('Phi, Theta, Psi [deg]')

        plt.figure(2)
        plt.clf()
        plt.plot(self.measure_PosX, self.measure_PosY, self.POS_X, self.POS_Y)
        plt.title('xy trajectory')
        plt.legend(['GPS','EKF'])

        # plt.figure(3)
        # plt.clf()
        # plt.plot(self.time,self.P_R)
        # plt.title('Covariance of Position')
        # plt.legend(['px','py','pz'])
        # plt.figure(4)
        # plt.clf()
        # plt.plot(self.time,self.P_R1)
        # plt.legend(['pxdot','pydot','pzdot'])
        # plt.title('Covariance of Velocities')
        # plt.figure(5)
        # plt.clf()
        # plt.plot(self.time,self.P_R2)
        # plt.title('Covariance of Quaternions')
        # plt.figure(6)
        # plt.clf()
        # plt.plot(self.time,self.Pos,self.time,self.measure_PosX,'r:', self.time,self.measure_PosY,'r:', self.time,self.measure_PosZ,'r:')
        # plt.legend(['X_ekf', 'Y_ekf', 'Z_ekf','Xgps','Ygps','Z_0'])
        # plt.title('Position')
        # plt.figure(7)
        # plt.clf()
        # plt.plot(self.time,self.Vel)
        # plt.title('vel x y z')
        # plt.figure(8)
        # plt.clf()
        # plt.plot(self.time,self.Quater)
        # plt.title('Quat')
        # plt.figure(9)
        # plt.clf()
        # plt.plot(self.time,self.P_angular,self.time,self.Q_angular,self.time,self.R_angular)
        # plt.title('OMEGA with Bias')
        # plt.legend(['p','q','r'])

        # plt.figure(10)
        # plt.clf()
        # plt.plot(self.time,self.Bias)
        # plt.title('Gyroscope and accelerometer Bias')
        # plt.legend(['bp','bq','br','bfx','bfy','bfz'])
                
        plt.ion()
        plt.show()
        plt.pause(0.0001)
        

def main(args=None):
    rclpy.init(args=args)

    ekf_node = ExtendedKalmanFilter()
    
    rclpy.spin(ekf_node)

   
    ekf_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
