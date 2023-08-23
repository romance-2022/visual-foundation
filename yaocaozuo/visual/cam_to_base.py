import cv2
import numpy as np
import glob
from math import *
import pandas as pd
import os
from scipy.spatial.transform import Rotation
import csv

from calibration import mtx
from calibration import dist


K=np.array(mtx,dtype=np.float64)#大恒相机内参

distCoeffs = dist
#dist_coeffs = np.array([0, 0, 0, 0], dtype=np.float32)



# 定义标定板中每个方格的尺寸
square_size = 30 # mm

# 定义棋盘格中每行、每列的角点个数
pattern_size = (8, 5)


#用于根据欧拉角计算旋转矩阵
def myRPY2R_robot(x, y, z):
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = Rz@Ry@Rx
    return R

#用于根据位姿计算变换矩阵T
def pose_robot(x, y, z, Tx, Ty, Tz):
    thetaX = x / 180 * pi
    thetaY = y / 180 * pi
    thetaZ = z / 180 * pi
    R = myRPY2R_robot(thetaX, thetaY, thetaZ)
    t = np.array([[Tx], [Ty], [Tz]])
    RT5 = np.column_stack([R, t])  # 列合并
    RT5 = np.row_stack((RT5, np.array([0,0,0,1])))
    # RT1=np.linalg.inv(RT1)
    return RT5


#从棋盘格图片得到相机外参,计算board to cam 变换矩阵
def get_RT_from_chessboard(pattern_size,K,square_size):
    '''
    :param img_path: 读取图片路径
    :param K: 相机内参

    :square_size: 单位棋盘格长度,mm
    :return: 相机外参
    '''

    
    # 准备标定板上角点的位置
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    #print(pattern_points)


    # Create arrays to store object points and image points from all the images
    objpoints_array = []
    imgpoints_array = []

    # Get a list of calibration images
    images = glob.glob('E:/reinforcement learning 4.3/visual/photos2/*.jpg')

    #计算board to cam 变换矩阵
    R_all_chess_to_cam_1=[]
    T_all_chess_to_cam_1=[]
    
    global good_photos
    good_photos = []

    # Loop over all calibration images
    for idx, fname in enumerate(images):
        # Load the image
        img = cv2.imread(fname)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('chessboard corners', gray)
        # cv2.waitKey(500)

        # Find the corners in the chessboard images
        ret, corners = cv2.findChessboardCorners(gray, (8, 5), None)
        #print(corners)

        # If corners are found, add object points and image points
        if ret:
            print(idx)
            good_photos.append(idx)
            print('找到角点:', fname)
            objpoints_array.append(pattern_points)
            imgpoints_array.append(corners)

            # 显示角点
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            cv2.imshow('chessboard corners', img)
            cv2.waitKey(500)

            retval,rvec,tvec  = cv2.solvePnP(pattern_points,corners, K, distCoeffs)
            # print(rvec.reshape((1,3)))
            # RT=np.column_stack((rvec,tvec))
            RT1=np.column_stack(((cv2.Rodrigues(rvec))[0],tvec))
            RT1 = np.row_stack((RT1, np.array([0, 0, 0, 1])))

            R_all_chess_to_cam_1.append(RT1[:3,:3])
            T_all_chess_to_cam_1.append(RT1[:3, 3].reshape((3,1)))

    cv2.destroyAllWindows()
    
    print('good photos are: ',good_photos)

    return R_all_chess_to_cam_1, T_all_chess_to_cam_1





#计算end to base变换矩阵
R_all_end_to_base_1=[]
T_all_end_to_base_1=[]
pos=[]
quaternion=[]
#good_photos=[0,3,4,7,8,9]

filename = "E:/reinforcement learning 4.3/visual/posandquar.csv" #从记录文件读取机器人位置+姿态=7个数

with open(filename, "r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:

        pos.append(row[0:3])
        quaternion.append(row[3:7])

pos = np.array(pos)
#print('positions: ', pos)
#print('quaternions: ', quaternion)


for i in good_photos:
    # 假设给定的四元数为 [w, x, y, z]
    # 创建旋转对象并转换为旋转矩阵
    r = Rotation.from_quat(quaternion[i])
    rotation_matrix = r.as_matrix()

    # 输出旋转矩阵
    #print(rotation_matrix)

    # Create 4x4 transformation matrix
    RT2 = np.zeros((4,4))
    RT2[:3, :3] = rotation_matrix
    RT2[:3, 3] = pos[i].T
    RT2[3, 3] = 1

    R_all_end_to_base_1.append(RT2[:3, :3])
    T_all_end_to_base_1.append(RT2[:3, 3].reshape((3, 1)))

print(T_all_end_to_base_1)



# 手眼标定
R_all_chess_to_cam_1, T_all_chess_to_cam_1 = get_RT_from_chessboard(pattern_size,K,square_size)

R,T=cv2.calibrateHandEye(R_all_end_to_base_1,T_all_end_to_base_1,R_all_chess_to_cam_1,T_all_chess_to_cam_1)
RT=np.column_stack((R,T))
RT = np.row_stack((RT, np.array([0, 0, 0, 1])))#即为cam to end变换矩阵
print('相机相对于末端的变换矩阵为：')
print(RT)





# #结果验证，原则上来说，每次结果相差较小 【这是眼在手内的！】
# for i in range(len(good_picture)):

#     RT_end_to_base=np.column_stack((R_all_end_to_base_1[i],T_all_end_to_base_1[i]))
#     RT_end_to_base=np.row_stack((RT_end_to_base,np.array([0,0,0,1])))
#     # print(RT_end_to_base)

#     RT_chess_to_cam=np.column_stack((R_all_chess_to_cam_1[i],T_all_chess_to_cam_1[i]))
#     RT_chess_to_cam=np.row_stack((RT_chess_to_cam,np.array([0,0,0,1])))
#     # print(RT_chess_to_cam)

#     RT_cam_to_end=np.column_stack((R,T))
#     RT_cam_to_end=np.row_stack((RT_cam_to_end,np.array([0,0,0,1])))
#     # print(RT_cam_to_end)

#     RT_chess_to_base=RT_end_to_base@RT_cam_to_end@RT_chess_to_cam#即为固定的棋盘格相对于机器人基坐标系位姿
#     RT_chess_to_base=np.linalg.inv(RT_chess_to_base) # @为矩阵乘法
#     print('第',i,'次')
#     print(RT_chess_to_base[:3,:])
#     print('')

