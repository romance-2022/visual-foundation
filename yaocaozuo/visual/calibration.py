import numpy as np
import cv2
import glob


# 定义标定板中每个方格的尺寸
square_size = 30 # mm

# 定义棋盘格中每行、每列的角点个数
pattern_size = (11, 9)

# 准备标定板上角点的位置
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

#print(pattern_points)


# Create arrays to store object points and image points from all the images
objpoints_array = []
imgpoints_array = []
good_photos=[]

# Get a list of calibration images
images = glob.glob('E:/reinforcement learning 4.3/visual/photos2/*.jpg')

# Loop over all calibration images
for idx, fname in enumerate(images):
    # Load the image
    img = cv2.imread(fname)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('chessboard corners', gray)
    # cv2.waitKey(500)

    # Find the corners in the chessboard images
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
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

cv2.destroyAllWindows()

print(good_photos)


# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_array, imgpoints_array, gray.shape[::-1], None, None)

# # Print the camera matrix and distortion coefficients
print("Camera matrix:\n", mtx) # 内参矩阵！！！
print("Distortion coefficients:\n", dist)
