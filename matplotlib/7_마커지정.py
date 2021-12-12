import matplotlib.pyplot as plt
import numpy as np

#. 기본 사용

# plt.plot([1, 2, 3, 4], [2, 3, 5, 10], 'bo') #. blue + circle
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# plt.show()

#. 선 + 마커
# # plt.plot([1, 2, 3, 4], [2, 3, 5, 10], 'bo-')    # 파란색 + 마커 + 실선
# plt.plot([1, 2, 3, 4], [2, 3, 5, 10], 'bo--')     # 파란색 + 마커 + 점선
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# plt.show()

#. marker 파라미터

plt.plot([4, 5, 6], marker="H")
plt.plot([3, 4, 5], marker="d")
plt.plot([2, 3, 4], marker="x")
plt.plot([1, 2, 3], marker=11)
plt.plot([0, 1, 2], marker='$Z$')
plt.show()