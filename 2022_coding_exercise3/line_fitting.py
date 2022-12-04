import numpy as np
import math
import matplotlib.pyplot as plt

# test data in polar coordinates
rho_test = np.array([[10, 11, 11.7, 13, 14, 15, 16, 17, 17, 17, 16.5,
17, 17, 16, 14.5, 14, 13]]).T
n = rho_test.shape[0]
theta_test = (math.pi/180)*np.linspace(0, 85, n).reshape(-1,1)

x = []
y = []
point_set = np.zeros((n, 2))
for i in range(n):
    x.append(rho_test[i] * math.cos(theta_test[i]))
    y.append(rho_test[i] * math.sin(theta_test[i]))
    point_set[i][0] = x[-1]
    point_set[i][1] = y[-1]
plt.scatter(x, y)

point_set = np.array(point_set)

point_set = np.concatenate((point_set, np.ones((n, 1))), axis=1)
line_set = []
# def merge():
print(point_set.shape)


def split(point_set, start, end, threshold):
    point_start = point_set[start]
    point_end = point_set[end]
    print("checking from", start, " to ", end)
    line = np.cross(point_start, point_end)
    if(end - start == 1):
        line_set.append([point_start[0], point_start[1], point_end[0], point_end[1]])
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
        split(point_set, start, max_ind, threshold)
        split(point_set, max_ind, end, threshold)
    else:
        line_set.append([point_start[0], point_start[1], point_end[0], point_end[1]])

split(point_set, 0, n-1, 1)
# line_set = np.array(line_set)
print(line_set)
for line in line_set:
    plt.plot([line[0], line[2]], [line[1], line[3]])

plt.show()