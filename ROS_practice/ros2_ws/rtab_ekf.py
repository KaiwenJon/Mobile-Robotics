import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

rtab_odom = np.loadtxt("./rtab1.txt")
ekf_odom = np.loadtxt("./ekf1.txt")

n_ekf = ekf_odom.shape[0]
n_rtab = rtab_odom.shape[0]
print(n_ekf)
print(n_rtab)
print(n_ekf/n_rtab)

total_sample = None
sum = 0
j = 0
for i in range(n_ekf):
    p_ekf = ekf_odom[i]
    if(i%5 < 4):
        p_rtab = rtab_odom[j]
    else:
        j += 1
        # print("jump")
    # print(p_ekf[0])
    # print(p_rtab[0])
    sum += (np.sum((p_ekf-p_rtab)**2))
RMSE = np.sqrt(sum/i)
print(RMSE)

# r = R.from_euler('z', -180, degrees=True)
# rtab_odom = r.apply(rtab_odom)


fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.set_aspect('equal')
# ax.set_proj_type('persp',focal_length=0.2) 
ax.scatter(ekf_odom[:, 0], ekf_odom[:, 1], ekf_odom[:, 2], c='b', marker='o', alpha=0.6, label = "ekf")
ax.scatter(rtab_odom[:, 0], rtab_odom[:, 1], rtab_odom[:, 2], c='r', marker='o', alpha=0.6, label = "rtab")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.legend()
plt.show()

