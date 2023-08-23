import numpy as np
from scipy.spatial.transform import Rotation as R



# 输入变换矩阵T
RT = np.array([[ 0.9511,  0.1364, -0.2774,  0.2   ],
              [-0.1621,  0.9667, -0.1961,  0.1   ],
              [ 0.2622,  0.2183,  0.9396,  0.3   ],
              [ 0.    ,  0.    ,  0.    ,  1.    ]])

# 从变换矩阵中提取位置
t = RT[:3, 3]

# 从变换矩阵中提取旋转矩阵
# 从旋转矩阵中计算四元数
r = R.from_matrix(RT[:3, :3])
q = r.as_quat()

print("Position: ", t)
#print('R', r)
print("Quaternion: ", q)