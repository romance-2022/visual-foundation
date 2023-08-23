# /usr/lib/python2.7/dist-packages/

import cv2
import numpy as np
import os

# 定义棋盘格尺寸
pattern_size = (12, 10)

# 定义每个格子的尺寸
square_size = 30

# 生成棋盘格内部的格子
board = np.zeros((pattern_size[1] * square_size, pattern_size[0] * square_size), np.uint8)
for i in range(pattern_size[1]):
    for j in range(pattern_size[0]):
        x = j * square_size
        y = i * square_size
        if (i+j) % 2 == 0:
            board[y:y+square_size, x:x+square_size] = 255

# 加上一圈白色边框
border_size = 30
border_color = 255
bordered_board = cv2.copyMakeBorder(board, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, None, value=border_color)

# 显示棋盘格
cv2.imshow("board", bordered_board)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 保存棋盘格图片
if not os.path.exists('/home/yang/ros_ws/src/baxter_examples/scripts/zjy/yaocaozuo/visual/photo2/'):
    os.mkdir('/home/yang/ros_ws/src/baxter_examples/scripts/zjy/yaocaozuo/visual/photo2/')
for i in range(10):
    filename = '/home/yang/ros_ws/src/baxter_examples/scripts/zjy/yaocaozuo/visual/photo2/chessboard{}.jpg'.format(i+1)
    cv2.imwrite(filename, bordered_board)

