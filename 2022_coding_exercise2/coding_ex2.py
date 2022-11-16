# Student name: 

import math
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
        # self.p = 0.0
        # self.q = 0.0
        # self.r = 0.0
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
        self.xhat = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.q1, self.q2, self.q3, self.q4, self.bp, self.bq, self.br, self.bfx, self.bfy, self.bfz]]).T

        self.rgps=np.array([-0.50, 0 ,0]) #This is the location of the GPS wrt CG, this is very important
        
        #noise params process noise (my gift to you :))
        self.Q = 100*np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.5, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
        #measurement noise
        #GPS position and velocity
        self.R = 1*np.diag([10, 10, 10, 2, 2, 2])
        
       
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
        
        self.timer_ekf = self.create_timer(self.dt, self.ekf_callback)
        self.timer_plot = self.create_timer(2, self.plot_data_callback)

        self.odom_pub = self.create_publisher(Odometry, 'odom', 10) #keep in mind how to declare publishers for next assignments
        self.timer_odom = self.create_timer(0.1, self.odom_callback) #It creates a timer to periodically publish the odometry.
        
        self.tf_broadcaster = TransformBroadcaster(self) # To broadcast the transformation between coordinate frames.

    
    def callback_imu(self,msg):
        #measurement vector = [p, q, r, fx, fy, fz, x, y, z, vx, vy, vz]
        # In practice, the IMU measurements should be filtered. In this coding exercise, we are just going to clip
        # the values of velocity and acceleration to keep them in physically possible intervals.
        self.measure[0] = np.clip(msg.angular_velocity.x,-5,5) #(-5,5)
        self.measure[1] = np.clip(msg.angular_velocity.y,-5,5) #..(-5,5)
        self.measure[2] = np.clip(msg.angular_velocity.z,-5,5)  #..(-5,5)
        self.measure[3] = np.clip(msg.linear_acceleration.x,-6,6)  #..(-6,6)
        self.measure[4] = np.clip(msg.linear_acceleration.y,-6,6)  #..(-6,6)
        self.measure[5] = np.clip(msg.linear_acceleration.z,-16,-4)  #..(-16,-4) 
 
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
        self.measure[9] = msg.data # ..
        self.measure[11] = 0.0 # vz

    def callback_gps_speed_north(self, msg):
        self.measure[10] = msg.data # vy

   
    def ekf_callback(self):
        #print("iteration:  ",self.loop_t)
        if (self.flag_lat and self.flag_lon):  #Trick  to sincronize rosbag with EKF
            self.ekf_function()
        else:
            print("Play the rosbag file...")

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
    
        self.q1 = self.quat[0,0] # w
        self.q2 = self.quat[1,0] # x
        self.q3 = self.quat[2,0] # y
        self.q4 = self.quat[3,0] # z
                
        # Rotation matrix: body to inertial frame
        self.R_bi = np.array([[pow(self.q1,2)+pow(self.q2,2)-pow(self.q3,2)-pow(self.q4,2), 2*(self.q2*self.q3-self.q1*self.q4), 2*(self.q2*self.q4+self.q1*self.q3)],
                          [2*(self.q2*self.q3+self.q1*self.q4), pow(self.q1,2)-pow(self.q2,2)+pow(self.q3,2)-pow(self.q4,2), 2*(self.q3*self.q4-self.q1*self.q2)],
                          [2*(self.q2*self.q4-self.q1*self.q3), 2*(self.q3*self.q4+self.q1*self.q2), pow(self.q1,2)-pow(self.q2,2)-pow(self.q3,2)+pow(self.q4,2)]])
        
            
        #Prediction step
        #First write out all the dots for all the states, e.g. pxdot, pydot, q1dot etc
       
        pxdot = self.xhat[3, 0]
        pydot = self.xhat[4, 0]
        pzdot = self.xhat[5, 0]
        vxdot, vydot, vzdot = np.matmul(self.R_bi, np.array([self.fx, self.fy, self.fz]).transpose())
        Omega = np.array([[0, self.p, self.q, self.r],
                    [-self.p, 0, -self.r, self.q],
                    [-self.q, self.r, 0, -self.p],
                    [-self.r, -self.q, self.p, 0]])
        
        q1dot, q2dot, q3dot, q4dot = -0.5*np.matmul(Omega, [self.q1, self.q2, self.q3, self.q4])
        bpdot = 0
        bqdot = 0
        brdot = 0
        bxdot = 0
        bydot = 0
        bzdot = 0
        # .. your code here
        
        #Now integrate Euler Integration for Process Updates and Covariance Updates
        # Euler works fine
        # Remember again the state vector [x y z vx vy vz q1 q2 q3 q4 bp bq br bx by bz]
        self.xhat[0,0] = self.xhat[0,0] + self.dt*pxdot
        self.xhat[1,0] = self.xhat[1,0] + self.dt*pydot # ..
        self.xhat[2,0] = self.xhat[2,0] + self.dt*pzdot # ..
        self.xhat[3,0] = self.xhat[3,0] + self.dt*vxdot # ..
        self.xhat[4,0] = self.xhat[4,0] + self.dt*vydot # ..
        self.xhat[5,0] = self.xhat[5,0] + self.dt*(vzdot - 9.801) # .. Do not forget Gravity (9.801 m/s2) 
        self.xhat[6,0] = self.xhat[6,0] + self.dt*q1dot # ..
        self.xhat[7,0] = self.xhat[7,0] + self.dt*q2dot # ..
        self.xhat[8,0] = self.xhat[8,0] + self.dt*q3dot # ..
        self.xhat[9,0] = self.xhat[9,0] + self.dt*q4dot # ..
        self.xhat[10,0] = self.xhat[10,0] + self.dt*bpdot
        self.xhat[11,0] = self.xhat[11,0] + self.dt*bqdot
        self.xhat[12,0] = self.xhat[12,0] + self.dt*brdot
        self.xhat[13,0] = self.xhat[13,0] + self.dt*bxdot
        self.xhat[14,0] = self.xhat[14,0] + self.dt*bydot
        self.xhat[15,0] = self.xhat[15,0] + self.dt*bzdot
        print("x ekf: ", self.xhat[0,0])
        print("y ekf: ", self.xhat[1,0])
        print("z ekf: ", self.xhat[2,0])
        print(vzdot)

        # Extract and normalize the quat    
        self.quat = np.array([[self.xhat[6,0], self.xhat[7,0], self.xhat[8,0], self.xhat[9,0]]]).T
        # .. Normailize quat
        self.quat = self.quat / LA.norm(self.quat) # code here. Uncomment this line
        #re-assign quat
        self.xhat[6,0] = self.quat[0,0]
        self.xhat[7,0] = self.quat[1,0]
        self.xhat[8,0] = self.quat[2,0]
        self.xhat[9,0] = self.quat[3,0]
        
                
        # Now write out all the partials to compute the transition matrix Phi
        #delV/delQ
        self.q1 = self.quat[0,0]
        self.q2 = self.quat[1,0]
        self.q3 = self.quat[2,0]
        self.q4 = self.quat[3,0]
        Fvq = 2*np.array([[(self.q1*self.fx - self.q4*self.fy + self.q3*self.fz),(self.q2*self.fx + self.q3*self.fy + self.q4*self.fz),(-self.q3*self.fx + self.q2*self.fy + self.q1*self.fz),(-self.q4*self.fx - self.q1*self.fy + self.q2*self.fz)],
            [(self.q4*self.fx + self.q1*self.fy - self.q2*self.fz),(self.q3*self.fx - self.q2*self.fy - self.q1*self.fz),(self.q2*self.fx + self.q3*self.fy + self.q4*self.fz),(self.q1*self.fx - self.q4*self.fy + self.q3*self.fz)],
            [(-self.q3*self.fx + self.q2*self.fy + self.q1*self.fz),(self.q4*self.fx + self.q1*self.fy - self.q2*self.fz),(-self.q1*self.fx + self.q4*self.fy - self.q3*self.fz),(self.q2*self.fx + self.q3*self.fy + self.q4*self.fz)]])
        #delV/del_abias
        
        Fvb = -self.R_bi # ..
        
        #delQ/delQ
        
        Fqq = -0.5*Omega # ..
     
        #delQ/del_gyrobias
        Fqb = 0.5*np.array([[self.q2, self.q3, self.q4],
                            [-self.q1, self.q4, -self.q3],
                            [-self.q4, -self.q1, self.q2],
                            [self.q3, -self.q2, -self.q1]]) # ..TODO check this sign
        # Now assemble the Transition matrix A
        
        z33 = np.zeros((3,3))
        z43 = np.zeros((4,3))
        z34 = np.zeros((3,4))
        z36 = np.zeros((3,6))
        i33 = np.identity(3)
        A = np.concatenate((np.concatenate((z33,i33,z34,z33,z33),axis=1),
                      np.concatenate((z33,z33,Fvq,z33,Fvb),axis=1),
                      np.concatenate((z43,z43,Fqq,Fqb,z43),axis=1),
                      np.concatenate((z33,z33,z34,z33,z33),axis=1),
                      np.concatenate((z33,z33,z34,z33,z33),axis=1)),axis=0) # ..
        
        #Propagate the error covariance matrix, I suggest using the continuous integration since Q, R are not discretized 
        # Pdot = A@P+P@A.transpose() + Q
        #P = P +Pdot*dt
        Pdot = A @ self.P + self.P @ A.transpose() + self.Q # .. 
        self.P = self.P + Pdot*self.dt # ..
        
        #Correction step
        #Get measurements 3 positions and 3 velocities from GPS
        self.z = np.array([[self.measure[6], self.measure[7], self.measure[8], self.measure[9], self.measure[10], self.measure[11]]]).T #x y z vx vy vz
    
        #Write out the measurement matrix linearization to get H
        
        # del v/del q
        Hvq = 2 * self.rgps[0] * np.array([[self.q3*self.q + self.q4*self.r, self.q4*self.q - self.q3*self.r, self.q1*self.q-self.q2*self.r, self.q2*self.q + self.q1*self.r],
                                           [-self.q2*self.q - self.q1*self.r, self.q2*self.r - self.q1*self.q, self.q4*self.q-self.q3*self.r,self.q3*self.q + self.q4*self.r],
                                           [ self.q1*self.q - self.q2*self.r,-self.q2*self.q - self.q1*self.r,-self.q3*self.q-self.q4*self.r,self.q4*self.q - self.q3*self.r]]) # ..
        
        #del P/del q
        Hxq = 2 * self.rgps[0] * np.array([[-self.q1,-self.q2,self.q3,self.q4],
                                           [-self.q4,-self.q3,-self.q2,-self.q1],
                                           [self.q3,-self.q4,self.q1,-self.q2]]) 
        
        # Assemble H
        H = np.concatenate((np.concatenate((i33,z33,Hxq,z36),axis=1),
                      np.concatenate((z33,i33,Hvq,z36),axis=1)),axis=0) # ..

        #Compute Kalman gain
        S = H @ self.P @ H.transpose() + self.R
        L = self.P @ H.transpose() @ LA.inv(S) # Kalman gain
        print("gain", LA.norm(L))
        #Perform xhat correction    xhat = xhat + L@(z-H@xhat)
        self.xhat = self.xhat + L @ (self.z - (H @ self.xhat)) # .. uncomment
        
        #propagate error covariance approximation P = (np.eye(16,16)-L@H)@P
        
        # self.P = self.P -L@S@L.transpose() # ..
        self.P = (np.eye(16,16)-L@H)@self.P

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
        
    def odom_callback(self):
        '''
        Compute the linear and angular velocity of the robot using the differential-drive robot kinematics
        Perform Euler integration to find the position x and y of the robot
        '''
        if (not self.flag_lat or not self.flag_lon):
            return
        dt = self.dt # DeltaT
        position = [self.POS_X[-1], self.POS_Y[-1], 0.0]
        if len(self.Quater) == 0:
            quater = [0.0, 0.0, 0.0, 0.0]
        else:
            quater = self.Quater[-1]
            quater = quater / LA.norm(quater)
        velocity = self.xhat[3, 0], self.xhat[4, 0], self.xhat[5, 0]
        angular_velocity = self.p, self.q, self.r
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
        odom.pose.pose.position.z = position[2] # ... 
        odom.pose.pose.orientation.x = quater[1]
        odom.pose.pose.orientation.y = quater[2] # ...
        odom.pose.pose.orientation.z = quater[3] # ...
        odom.pose.pose.orientation.w = quater[0] # ...

        # set the velocities. Uncomment next lines
        odom.child_frame_id = child_frame_id
        odom.twist.twist.linear.x = velocity[0] # ...
        odom.twist.twist.linear.y = velocity[1] # ...
        odom.twist.twist.linear.z = velocity[2] # ...
        odom.twist.twist.angular.x = angular_velocity[0] # ...
        odom.twist.twist.angular.y = angular_velocity[1] # ...
        odom.twist.twist.angular.z = angular_velocity[2] # ...

        self.odom_pub.publish(odom)

        
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

        t.transform.rotation.x = quater[1] # ...
        t.transform.rotation.y = quater[2] # ...
        t.transform.rotation.z = quater[3] # ...
        t.transform.rotation.w = quater[0] # ...

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)
def main(args=None):
    rclpy.init(args=args)

    ekf_node = ExtendedKalmanFilter()
    
    rclpy.spin(ekf_node)

   
    ekf_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()