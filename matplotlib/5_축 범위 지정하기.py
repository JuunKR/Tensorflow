import matplotlib.pyplot as plt
import numpy as np


#. xlim(), ylim()

# plt.plot([1, 2, 3, 4], [2, 3, 5, 10])
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# plt.xlim([0, 5])      # X축의 범위: [xmin, xmax]
# plt.ylim([0, 20])     # Y축의 범위: [ymin, ymax]

# plt.show()


#. axis()

# plt.plot([1, 2, 3, 4], [2, 3, 5, 10])
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# plt.axis([0, 5, 0, 20])  # X, Y축의 범위: [xmin, xmax, ymin, ymax]

# plt.show()

#옵션
# plt.plot([1, 2, 3, 4], [2, 3, 5, 10])
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# # plt.axis('square')
# plt.axis('equal')
# #. 'on' | 'off' | 'equal' | 'scaled' | 'tight' | 'auto' | 'normal' | 'image' | 'square'

# plt.show()

#. 축 범위 얻기

plt.plot([1, 2, 3, 4], [2, 3, 5, 10])
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

x_range, y_range = plt.xlim(), plt.ylim()
print(x_range, y_range)
#. xlim(), ylim() 함수는 그래프 영역에 표시되는 X축, Y축의 범위를 각각 반환
# (0.85, 4.15) (1.6, 10.4)

axis_range = plt.axis('scaled')
print(axis_range)
#. axis() 함수는 그래프 영역에 표시되는 X, Y축의 범위를 반환
# (0.85, 4.15, 1.6, 10.4)

plt.show()