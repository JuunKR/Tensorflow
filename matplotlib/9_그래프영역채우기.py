import matplotlib.pyplot as plt
import numpy as np


#. fill_between()

# x = [1, 2, 3, 4]
# y = [2, 3, 5, 10]

# plt.plot(x, y)
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# plt.fill_between(x[1:3], y[1:3], alpha=0.5)

# plt.show()

#. fill_beteenx()

# x = [1, 2, 3, 4]
# y = [2, 3, 5, 10]

# plt.plot(x, y)
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# plt.fill_betweenx(y[2:4], x[2:4], alpha=0.5)

# plt.show()

#.두 그래프 사이 영역 채우기

# x = [1, 2, 3, 4]
# y1 = [2, 3, 5, 10]
# y2 = [1, 2, 4, 8]

# plt.plot(x, y1)
# plt.plot(x, y2)
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# plt.fill_between(x[1:3], y1[1:3], y2[1:3], color='lightgray', alpha=0.5)

# plt.show()

#. 다각형 영역 채우기 - fill()

x = [1, 2, 3, 4]
y1 = [2, 3, 5, 10]
y2 = [1, 2, 4, 8]

plt.plot(x, y1)
plt.plot(x, y2)
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.fill([1.9, 1.9, 3.1, 3.1], [1.0, 4.0, 6.0, 3.0], color='lightgray', alpha=0.5)

plt.show()