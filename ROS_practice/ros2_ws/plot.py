import matplotlib.pyplot as plt

f = open("results_part1_wheel.txt", "r")
x = []
y = []
cnt = 0
for line in f:
  x.append(round(float(line.split()[1]), 3))
  y.append(round(float(line.split()[2]), 3))
#   plt.scatter(x[-1], y[-1])
#   cnt += 1
#   if(cnt > 500):
#     break
# print(x)
# print(y)
plt.plot(x, y)
plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
plt.show()
