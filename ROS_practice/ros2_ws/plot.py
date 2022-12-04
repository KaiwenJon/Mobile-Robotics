import numpy as np
import math
import time
import matplotlib.pyplot as plt

# ranges = np.load('laser.npy')
# n = len(ranges)
# point_set = np.zeros((n, 3))
# for i in range(n):
#     point_set[i][0] = ranges[i] * math.cos(-math.pi/4 + math.pi*3/2/(n-1)*i) + 0
#     point_set[i][1] = ranges[i] * math.sin(-math.pi/4 + math.pi*3/2/(n-1)*i) + 0
#     point_set[i][2] = 1

# x = point_set[:, 0]
# y = point_set[:, 1]
# plt.scatter(x, y)
robot_pos = np.load("robot_pos.npy")

t1 = time.time()
point_set = np.load('laser.npy')
print(point_set)
x = point_set[:, 0]
y = point_set[:, 1]
plt.scatter(x, y)
plt.scatter(robot_pos[0], robot_pos[1])
n = point_set.shape[0]
line_set = []
line_set_far = []
line_set_margin = []
def split(point_set, start, end, threshold):

    print("checking from", start, " to ", end)
    point_start = point_set[start]
    point_end = point_set[end]
    print(point_start, point_end)
    line = np.cross(point_start, point_end)
    if(end - start < 5):
        # line_set.append([point_start[0], point_start[1], point_end[0], point_end[1]])
        line_set_far.append([point_start[0], point_start[1], point_end[0], point_end[1]])
        return
    q = np.sqrt(np.sum(line[0:2]**2))
    print(q)
    max_d = -float('inf')
    max_ind = -1
    cnt = 0
    for i in range(start+1, end):
        dist = np.abs(np.sum(point_set[i] * line)) / q
        # print(dist)
        if(dist >= max_d):
            max_d = dist
            max_ind = i
            cnt += 1
    if(max_d > threshold):
        split(point_set, start, max_ind-1, threshold)
        split(point_set, max_ind+1, end, threshold)
    else:
        start_d = np.sqrt(np.sum((point_start - robot_pos))**2)
        end_d = np.sqrt(np.sum((point_end - robot_pos))**2)
        if (start_d > 20 or end_d > 20):
          line_set_margin.append([point_start[0], point_start[1], point_end[0], point_end[1]])
        else:
          line_set.append([point_start[0], point_start[1], point_end[0], point_end[1]])
split(point_set, 20, n-1, 0.08)
t2 = time.time()
print(t2-t1)
for line in line_set:
    plt.plot([line[0], line[2]], [line[1], line[3]], color='r')
for line in line_set_far:
    plt.plot([line[0], line[2]], [line[1], line[3]], color='g')
for line in line_set_margin:
    plt.plot([line[0], line[2]], [line[1], line[3]], color='b')    

plt.show()
        