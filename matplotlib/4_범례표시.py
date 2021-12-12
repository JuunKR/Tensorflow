import matplotlib.pyplot as plt
import numpy as np


#. 기본사용

# plt.plot([1, 2, 3, 4], [2, 3, 5, 10], label='Price ($)')
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# plt.legend()

# plt.show()

#. 위치지정하기

# plt.plot([1, 2, 3, 4], [2, 3, 5, 10], label='Price ($)')
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# # plt.legend(loc=(0.0, 0.0))
# plt.legend(loc=(0.5, 0.5))
# # plt.legend(loc=(1.0, 1.0))
# # plt.legend(loc='lower right')

# plt.show()

#. 열개수 지정하기

# plt.plot([1, 2, 3, 4], [2, 3, 5, 10], label='Price ($)')
# plt.plot([1, 2, 3, 4], [3, 5, 9, 7], label='Demand (#)')
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# # plt.legend(loc='best')          # ncol = 1 디폴트
# plt.legend(loc='best', ncol=2)    # ncol = 2

# plt.show()

#. 폰트크기 지정
# plt.plot([1, 2, 3, 4], [2, 3, 5, 10], label='Price ($)')
# plt.plot([1, 2, 3, 4], [3, 5, 9, 7], label='Demand (#)')
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# # plt.legend(loc='best')
# plt.legend(loc='best', ncol=2, fontsize=14)

# plt.show()

#. 폰트 테두리 꾸미기
plt.plot([1, 2, 3, 4], [2, 3, 5, 10], label='Price ($)')
plt.plot([1, 2, 3, 4], [3, 5, 9, 7], label='Demand (#)')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
# plt.legend(loc='best')
plt.legend(loc='best', ncol=2, fontsize=14, frameon=True, shadow=True)
# facecolor, edgecolor, borderpad, labelspacing 다른파라미터
plt.show()