#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import scipy.io as sio 
import matplotlib.pyplot as plt
import numpy as np
import rospy
import baxter_interface
import time
import thread
import math
from ntpath import join
from re import S
from sre_constants import SRE_FLAG_DEBUG
from baxter_pykdl import baxter_kinematics 
from math import sin, cos, sqrt
from std_msgs.msg import (
    UInt16
)
from baxter_interface import CHECK_VERSION
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from omni_msgs.msg import OmniState, OmniButtonEvent
from std_msgs.msg import Header
from baxter_core_msgs.srv import (
    SolvePositionIK,
    # SolForce_ReceivervePositionIKRequest,
    SolvePositionIKRequest,
)
from touchx_hhf_1211 import ik_test
from force_receiver_hhf import Force_Receiver
from scipy.spatial.transform import Rotation
from math import pi
from sensor_msgs.msg import JointState
from  baxter_core_msgs.msg import SEAJointState
class SLDF(object):
    def __init__(self):
        self.time_step = 0.01
        self.count = 0
        # Baxter
        self._left_arm = baxter_interface.limb.Limb("left")
        self.left_init_pos = np.array([0.0, 0.0, 0.0])
        self.left_cur_pos = np.array([0.0, 0.0, 0.0])
        self.left_tool_init = np.array([0.0, 0.0, 0.0])
        self.left_tool_pos= np.array([0.0, 0.0, 0.0])
        self.left_cur_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.left_init_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.sub_torque = rospy.Subscriber('/robot/limb/left/gravity_compensation_torques', SEAJointState, self.getTorque)

         
        self._right_arm = baxter_interface.limb.Limb("right")
        self.right_init_pos = np.array([0.0, 0.0, 0.0])
        self.right_cur_pos = np.array([0.0, 0.0, 0.0])
        self.right_tool_init = np.array([0.0, 0.0, 0.0])
        self.right_tool_pos = np.array([0.0, 0.0, 0.0])
        self.right_cur_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.right_init_quat = np.array([0.0, 0.0, 0.0, 1.0])

        # TouchX
        self.roll = 0  # 记录最后一个关节的角度
        self.touchx_pos = np.array([0.0, 0.0, 0.0])
        self.touchx_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.touchx_init_pos = np.array([0.0, 0.0, 0.0])
        self.touchx_init_quat = np.array([0.0, 0.0, 0.0, 1.0])  # 进去flipping任务时，touchx的姿态
        self.touchx_button_count = 0
        self.sub_state = rospy.Subscriber('/phantom/state', OmniState, self.getState)
        self.sub_button = rospy.Subscriber('/phantom/button', OmniButtonEvent, self.get_touchx_button_state)
        self.sub_joints = rospy.Subscriber('phantom/joint_states', JointState, self.getJoints)
        # print "touchx's position in __init__: ", self.touchx_pos

        # Box
        self.box_height = 0
        self.box_length = 0
        self.box_point = np.array([0, 0, 0]) # 这里需要后面实验标定
        
        self.flag = False
        self.delta_theta = 0
        self.theta_init = 0
        self.theta_cur = 0
        self.torque = [[0.],[0.],[0.],[0.],[0.],[0.],[0.]]
        self.box_theta = 0
        self.m = 0.3154
        self.g = 9.8

        # ATI mini45
        self.force_x = 0
        self.force_y = 0
        self.force_z = 0
        self.Kp = 25
        self.Kd = 2
        self.fd = 2
        self.xe = 0
        self.dxe = 0
        self.dt = 0.001

        # 坐标转换
        self.rotation_tb2bb = np.matrix([[0,0,1],[1,0,0],[0,1,0]]) #touchx base 相对于baxter base的矩阵
        self.rotation_te2tb = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
        self.rotation_be2bb = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
        self.rotation_te2be = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
        self.rotation_be2bb_right = np.matrix([[1,0,0],[0,1,0],[0,0,1]]) #baxter右臂姿态

        # 画图
        self.forcez = [0]
        self.xe_list = [0]
        self.dxe_list = [0]
        self.fd_list = [0]

    # 四元数转换成旋转矩阵
    def quat2R(self, quat):
        x = quat[0]
        y = quat[1]
        z = quat[2]
        w = quat[3]
        x2 = x**2
        y2 = y**2
        z2 = z**2
        r11 = 1-2*y2-2*z2
        r12 = 2*(x*y-z*w)
        r13 = 2*(x*z+y*w)
        r21 = 2*(x*y+z*w)
        r22 = 1-2*x2-2*z2
        r23 = 2*(y*z-x*w)
        r31 = 2*(x*z-y*w)
        r32 = 2*(y*z+x*w)
        r33 = 1-2*x2-2*y2
        return np.array([[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]])  

    # 旋转矩阵转四元数
    def R2quat(self, r):
        four_w_squared = 1 + r[0,0] + r[1,1] + r[2,2]
        four_x_squared = 1 + r[0,0] - r[1,1] - r[2,2]
        four_y_squared = 1 - r[0,0] + r[1,1] - r[2,2]
        four_z_squared = 1 - r[0,0] - r[1,1] + r[2,2]

        if four_w_squared > four_x_squared and four_w_squared > four_y_squared and four_w_squared > four_z_squared:
            w = sqrt(four_w_squared) / 2.0
            x = (r[2,1] - r[1,2]) / (4.0 * w)
            y = (r[0,2] - r[2,0]) / (4.0 * w)
            z = (r[1,0] - r[0,1]) / (4.0 * w)
            # print('first w')
        elif four_x_squared > four_w_squared and four_x_squared > four_y_squared and four_x_squared > four_z_squared:
            x = sqrt(four_x_squared) / 2.0
            w = (r[2,1] - r[1,2]) / (4.0 * x)
            y = (r[0,1] + r[1,0]) / (4.0 * x)
            z = (r[0,2] + r[2,0]) / (4.0 * x)
            # print('first x')
        elif four_y_squared > four_x_squared and four_y_squared > four_z_squared and four_y_squared > four_w_squared:
            y = sqrt(four_y_squared) / 2.0
            w = (r[0,2] - r[2,0]) / (4.0 * y)
            x = (r[0,1] + r[1,0]) / (4.0 * y)
            z = (r[1,2] + r[2,1]) / (4.0 * y)
            # print('first y')
        else:            
            z = sqrt(four_z_squared) / 2.0
            w = (r[1,0] - r[0,1]) / (4.0 * z)
            x = (r[0,2] + r[2,0]) / (4.0 * z)
            y = (r[1,2] + r[2,1]) / (4.0 * z)
            # print('first z')
        return [x, y, z, w]

    # 将Baxter的末端位置转移到夹具末端
    def baxter_end_forward_left(self, pos):
        cur_rot =  self.quat2R(self.left_cur_quat)
        delta_pos = np.dot(cur_rot, np.array([0, 0, 0.04]).T).T
        return pos + delta_pos
    def baxter_tool_afterward_left(self,pos):
        cur_rot =  self.quat2R(self.left_cur_quat)
        delta_pos = np.dot(cur_rot, np.array([0, 0, 0.04]).T).T
        return pos - delta_pos
    def baxter_tool_afterward_right(self,pos):
        cur_rot =  self.quat2R(self.right_cur_quat)
        delta_pos = np.dot(cur_rot, np.array([0, 0, 0.039]).T).T
        return pos - delta_pos

    # 接收TouchX的数据
    def getState(self, state_msg):
        pos = state_msg.pose.position
        self.touchx_pos = np.array([pos.x, pos.y, pos.z])/1000
        quat = state_msg.pose.orientation
        self.touchx_quat = np.array([quat.x, quat.y, quat.z, quat.w])
    def get_touchx_button_state(self, touchx_button_state):
        if touchx_button_state.grey_button==1:
            self.touchx_button_count += 1
            print 'grey_button=1',str(self.touchx_button_count)
        else:
            print 'self.touchx_button_count',self.touchx_button_count
            pass
    def getJoints(self, joints_msg):
        self.roll = joints_msg.position[5]
        # print "cur_roll: ", roll/pi*180     
    def getTorque(self, torque_msg):
        gravity = torque_msg.gravity_model_effort
        self.torque1 = gravity[0]
        self.torque2 = gravity[1]
        self.torque3 = gravity[2]
        self.torque4 = gravity[3]
        self.torque5 = gravity[4]
        self.torque6 = gravity[5]
        self.torque7 = gravity[6]
        

    # 计算向量的夹角
    def dot_product_angle(self,v1, v2):
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            print("Zero magnitude vector!")
        else:
            vector_dot_product = np.dot(v1, v2)
            arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            # angle = np.degrees(arccos)
            angle = arccos
            return angle
        return 0

    def teleoperation(self):
        # 1. 用TouchX控制Baxter左臂
        self.rotation_te2tb = self.quat2R(self.touchx_quat)
        self.rotation_be2bb = self.rotation_tb2bb * self.rotation_te2tb * self.rotation_te2be.T
        self.left_cur_quat = self.R2quat(self.rotation_be2bb)
        touchx_delta_pos = self.touchx_pos - self.touchx_init_pos
        baxter_delta_pos = np.dot(np.array(self.rotation_tb2bb), touchx_delta_pos.T).T
        
        self.left_cur_pos = self.left_init_pos + baxter_delta_pos

        baxter_left_joints = ik_test("left",self.left_cur_pos, self.left_cur_quat)
        if not baxter_left_joints== None:
                    self._left_arm.set_joint_positions(baxter_left_joints)
                    time.sleep(0.01)            
        else:
            print "not valid pose of left arm(teleoperation)..."
        # print '++++++++++++++++++++++++++++++++++++++++++++++++++'
        # print 'touchx_init_pos: ', self.touchx_init_pos
        # print 'touchx_pos: ', self.touchx_pos
        # print 'touchx_delta_pos: ', touchx_delta_pos
        # print 'baxter_delta_pos: ', baxter_delta_pos
        # print 'baxter_pos: ', self.left_cur_pos
        # print 'baxter_quat: ', self.quat2R(self.left_cur_quat)
        # print '--------------------------------------------------'

    def controlRight(self):
        ## 2. 根据左臂确定右臂
        #  2.1 右臂姿态确定
        dR = np.matrix([[1, 0, 0],
                    [0, cos(math.pi/2), -sin(math.pi/2)],
                    [0, sin(math.pi/2), cos(math.pi/2)]])

        self.rotation_be2bb_right = dR * self.rotation_be2bb
        self.right_cur_quat = self.R2quat(self.rotation_be2bb_right)
        #  2.2 右臂位置确定
        # 进入Flipping任务初始化
        # 计算右臂位置（需要提前定好 self.left_tool_init 和 self.right_tool_init）
        self.left_tool_pos = self.baxter_end_forward_left(self.left_cur_pos)
        left_init_inbox = self.left_tool_init - self.box_point  # 翻转开始时，左臂夹具在定点下的坐标
        left_cur_inbox = self.left_tool_pos - self.box_point    # 左臂夹具在定点下的坐标
        left_cur_inbox[0] = left_init_inbox[0]
        delta_angle = self.dot_product_angle(left_cur_inbox, left_init_inbox)
        delta_rot = np.matrix([[1, 0, 0],
                    [0, cos(-delta_angle), -sin(-delta_angle)],
                    [0, sin(-delta_angle), cos(-delta_angle)]])
        right_init_inbox = self.right_tool_init - self.box_point
        right_cur_inbox = np.dot(np.array(delta_rot), right_init_inbox.T).T
        self.right_tool_pos = right_cur_inbox + self.box_point
        self.right_cur_pos = self.baxter_tool_afterward_right(self.right_tool_pos)
        self.right_cur_pos[0] = self.left_cur_pos[0]
        baxter_right_joints = ik_test('right', self.right_cur_pos, self.right_cur_quat)
        if not baxter_right_joints== None:
                    self._right_arm.set_joint_positions(baxter_right_joints)
                    time.sleep(0.01)            
        else:
            print '============================================='
            print 'right_cur_pos', self.right_cur_pos
            print 'right_cur_quat: ', self.right_cur_quat
            print '============================================='
            print "not valid pose of right arm(controlRight)..."

    def OLC_tele(self):
        # 记得按按钮的时候更新touchx_init_quat以及theta_init
        # 已知：left_tool_init和right_tool_init，求left_cur_pos和right_cur_pos
        # 已知：left_init_quat和right_init_quat，求left_cur_quat和right_cur_quat
        # 根据touchx的姿态变化，确定箱子的转动角度，从而计算双臂的位置
        # 1. 计算touchx的姿态变化
        self.rotation_te2tb = self.quat2R(self.touchx_quat)
        rotation_te2tb_init = self.quat2R(self.touchx_init_quat)
        delta_R = rotation_te2tb_init.T * self.rotation_te2tb
        # delta_touchx_rot = Rotation.from_matrix(delta_R) # 旋转矩阵转换成旋转对象
        delta_touchx_rot = Rotation.from_dcm(delta_R) # 旋转矩阵转换成旋转对象
        axis_angle = delta_touchx_rot.as_rotvec() #旋转对象转换成旋转矢量
        # print "rotation matrix R: \n", delta_R
        touchx_z_axis = np.array([0,0,1])
        self.delta_theta = self.roll - self.theta_init + self.theta_cur
        self.box_point[0] = self.left_tool_init[0]
        
        # 2.求left_cur_pos和right_cur_pos
        left_init_inbox = self.left_tool_init - self.box_point
        right_init_inbox = self.right_tool_init - self. box_point
        dR = np.matrix([[1, 0, 0],
                    [0, cos(self.delta_theta), -sin(self.delta_theta)],
                    [0, sin(self.delta_theta), cos(self.delta_theta)]])

        left_cur_inbox = np.dot(np.array(dR), left_init_inbox.T).T
        self.left_tool_pos = left_cur_inbox + self.box_point
        self.left_cur_pos = self.baxter_tool_afterward_left(self.left_tool_pos)

        right_cur_inbox = np.dot(np.array(dR), right_init_inbox.T).T
        self.right_tool_pos = right_cur_inbox + self.box_point
        self.right_cur_pos = self.baxter_tool_afterward_right(self.right_tool_pos)


        # 3.求left_cur_quat和right_cur_quat
        left_init_rot = self.quat2R(self.left_init_quat)
        right_init_rot = self.quat2R(self.right_init_quat)
        self.left_cur_quat = self.R2quat(np.dot(np.array(dR),left_init_rot))
        self.right_cur_quat = self.R2quat(np.dot(np.array(dR),right_init_rot))

        # 4.力控
        # self.fd = self.m*self.g*sqrt(self.box_height*self.box_height+self.box_length*self.box_length)*sin(self.box_theta+self.delta_theta)
        self.fd_list.append(self.fd)
        self.forcez.append(-self.force_z)
        self.dxe = ((self.fd+self.force_z) - self.Kp * self.xe)/self.Kd
        self.xe += self.dxe * self.dt
        delta_pos = np.dot(self.quat2R(self.right_cur_quat), np.array([0.,0.,self.xe]).T)
        print '****************************************************'
        print 'delta pos: ', delta_pos
        print 'xe: ', self.xe   
        print 'fz: ', self.force_z
        print '****************************************************'
        # self.right_cur_pos = delta_pos + self.right_pos0
        self.right_cur_pos += delta_pos
        self.xe_list.append(self.xe)
        self.dxe_list.append(self.dxe)
                
        # # 驱动Baxter
        left_joints = ik_test('left', self.left_cur_pos, self.left_cur_quat)
        # self._left_arm.move_to_joint_positions(left_joints)
        if not left_joints== None:
            self._left_arm.set_joint_positions(left_joints)
            time.sleep(0.01)            
        else:
            print "not valid pose..."

        right_joints = ik_test('right', self.right_cur_pos, self.right_cur_quat)
        # self._right_arm.move_to_joint_positions(right_joints)
        if right_joints == None:
                print "No valid pose of right arm when starting to flip"
        else:
            self._right_arm.set_joint_positions(right_joints) 

        
    def ready_enter_flipping(self):
        # 进入任务的标志确定
        # if abs(self.force_z) > 5:
        #     init_time = time.time()
        #     cur_time = time.time()
        #     while cur_time-init_time < 0.1:
        #         print('+++++++++++++++++++++++++++self.force_z:',self.force_z)
        #         if abs(self.force_z) > 5:
                    
        #             self.flag = True
        #         else:
        #             self.flag = False
        #             break
        #         cur_time = time.time()
        # else:
        #     self.flag = False
        if self.touchx_button_count >= 9:
            self.flag = True
        
        # 进入翻转任务前的初始化
        if self.flag:
            self.theta_init = self.roll
            self.left_tool_init = self.baxter_end_forward_left(self.left_cur_pos)
            self.left_init_quat = self.left_cur_quat
            # 给定右臂的起始位置，还需要进一步确定方法
            self.right_tool_init[0] = self.left_tool_init[0]
            self.right_tool_init[1] = self.box_point[1] - self.box_length - 0.03
            self.right_tool_init[2] = self.box_point[2] + self.box_height - 0.04
            self.right_cur_pos = self.baxter_tool_afterward_right(self.right_tool_init)
            self.right_pos0 = self.right_cur_pos
            self.right_init_quat = self.right_cur_quat
            dR = np.matrix([[1, 0, 0],
                    [0, cos(math.pi/2), -sin(math.pi/2)],
                    [0, sin(math.pi/2), cos(math.pi/2)]])
            self.rotation_be2bb_right = dR * self.rotation_be2bb
            self.right_cur_quat = self.R2quat(self.rotation_be2bb_right)
            self.touchx_init_quat = self.touchx_quat # 确定起始的touchx姿态，用于后续计算旋转角度
            # self.right_cur_quat = np.array([0,0.7071,0.7071,0]) #用于测试力控
            
            i = 45
            while i==0:
                print '(Ready flipping)right_cur_pos: ', self.right_cur_pos
                i -= 1
            baxter_right_init_joints = ik_test('right', self.right_cur_pos, self.right_cur_quat)
            if baxter_right_init_joints == None:
                print "No valid pose of right arm when starting to flip"
            else:
                self._right_arm.move_to_joint_positions(baxter_right_init_joints)

  
    def run(self):
        self.pre_init()  #指定一些状态和初始位置
        my_force_receiver = Force_Receiver()
        my_force_receiver.start()
        while not rospy.is_shutdown():
            armforce = my_force_receiver.data
            self.force_x = armforce[0]
            self.force_y = armforce[1]
            self.force_z = armforce[2]

            # 检测按钮，奇数控制，偶数停止（未记录数据）
            if self.touchx_button_count%2 == 1:
                if not self.flag:
                    self.teleoperation()
                    print 'flag: ', self.flag
                    self.ready_enter_flipping()
                if self.flag:
                    if abs(self.delta_theta) > pi/2:
                        break
                    self.OLC_tele()
                    # self.controlRight()
                self.count += 1
                print "count=", self.count
        
            else:
                # 重新标定touchx与Baxter末端的变换矩阵
                print "Pause..."
                current_pose = self._left_arm.endpoint_pose()['position']
                current_quat = self._left_arm.endpoint_pose()['orientation']
                self.rotation_te2tb = self.quat2R(self.touchx_quat)
                self.rotation_be2bb = self.quat2R(np.array([current_quat.x, current_quat.y, current_quat.z, current_quat.w]))
                self.left_init_pos = current_pose
                self.touchx_init_pos = self.touchx_pos
                self.rotation_te2be = self.rotation_be2bb.T * self.rotation_tb2bb * self.rotation_te2tb
                self.touchx_init_quat = self.touchx_quat 
                self.theta_init = self.roll
                self.theta_cur = self.delta_theta
                # self.rotation_be2bb = self.rotation_tb2bb * self.rotation_te2tb * self.rotation_te2be.T
            
            if self.count > 50000:
                break
        print 'troque: ', self.torque[0]
        save_path = '/home/yang/ros_ws/src/baxter_examples/scripts/haifeng/SLDF'
        np.savetxt(save_path + 'torque.txt', self.torque[0], delimiter = '\n')
        print 'data save'
        my_force_receiver.stop()
        plt.figure()
        print 'forcez.len: ', len(self.forcez)
        t = range(0,len(self.forcez))
        fd = [-self.fd] * len(self.forcez)
        plt.plot(t,self.forcez, label='force')
        # plt.plot(t,fd, label='fd')
        plt.plot(t,self.fd_list,label='fd')
        # plt.plot(t,self.dxe_list, label='dxe')
        plt.legend()
        plt.show()
        
        
    def pre_init(self):
        print 'preparing...'
        # 左右臂位姿
        self.left_cur_pos = np.array([0.57, 0.4, 0.16])
        self.left_cur_quat = np.array([0,0.7071,-0.7071,0])
        self.left_init_pos = self.left_cur_pos
        left_init_joints = ik_test('left', self.left_cur_pos, self.left_cur_quat)
        self._left_arm.move_to_joint_positions(left_init_joints)

        self.right_cur_pos = np.array([0.57, -0.4, 0.16])
        self.right_cur_quat = np.array([0,0.7071,0.7071,0])
        right_init_joints = ik_test('right', self.right_cur_pos, self.right_cur_quat)
        self._right_arm.move_to_joint_positions(right_init_joints)

        time.sleep(0.1) #确保touchx的数据已经更新
        self.touchx_init_pos = self.touchx_pos
        self.touchx_init_quat = self.touchx_quat
        # print 'touchx_init_pos: ', self.touchx_init_pos
        print 'touchx_init_quat: ', self.touchx_init_quat
        # time.sleep(0.1)
        # Box
        self.box_height = 0.355
        self.box_length = 0.258
        self.box_theta = np.arctan(self.box_height/self.box_length)
        self.box_point = np.array([0.6213264456813667, 0.02, -0.145]) # 这里需要后面实验标定
        self.flag = False

        # 末端姿态对应
        quat = self._left_arm.endpoint_pose()['orientation']
        self.rotation_be2bb = self.quat2R(np.array([quat.x, quat.y, quat.z, quat.w]))
        self.rotation_te2tb = self.quat2R(self.touchx_quat)
        print 'touchx_quat: ', self.touchx_quat
        self.rotation_te2be = self.rotation_be2bb.T * self.rotation_tb2bb * self.rotation_te2tb
        # print '************************************'
        # print 'rotation_te2tb: '
        # print self.rotation_te2tb
        # print 'rotation_tb2bb: '
        # print self.rotation_tb2bb
        # print 'rotation_be2bb: '
        # print self.rotation_be2bb
        # print 'rotation_te2be: '
        # print self.rotation_te2be
        # print '************************************'
        # print 'preparing is end...' 
        return 0

    def clean_shutdown(self):
        print("\nExiting experiment...")
        self.set_neutral()
        return True
    
    def set_neutral(self):
        """
        Sets both arms back into a neutral pose.
        """
        print("Moving to neutral left pose...")
        self.left_cur_pos = np.array([0.57, 0.4, 0.25])
        self.left_cur_quat = np.array([0,0.7071,-0.7071,0])
        left_init_joints = ik_test('left', self.left_cur_pos, self.left_cur_quat)
        if not left_init_joints== None:
                    self._left_arm.move_to_joint_positions(left_init_joints)
                    # time.sleep(0.01)      
        else:
            print "not valid pose..."

        print("Moving to neutral right pose...")
        self.right_cur_pos = np.array([0.57, -0.4, 0.16])
        self.right_cur_quat = np.array([0,0.7071,0.7071,0])
        self.right_cur_pos = np.array([0.57, -0.4, 0.25])
        self.right_cur_quat = np.array([0,0.7071,0.7071,0])
        right_init_joints = ik_test('right', self.right_cur_pos, self.right_cur_quat)
        if not right_init_joints== None:
                    self._right_arm.move_to_joint_positions(right_init_joints)
                    # time.sleep(0.01)
        else:
            print "not valid pose..."
         
def main():
    print("Initializing node... ")
    rospy.init_node("SLDF")

    sldf = SLDF()
    rospy.on_shutdown(sldf.clean_shutdown)
    sldf.run()

if __name__ == '__main__':
    main()


    
