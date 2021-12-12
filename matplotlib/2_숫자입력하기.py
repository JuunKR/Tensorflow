import matplotlib.pyplot as plt
import numpy as np


#. 기본 사용

# plt.plot([2, 3, 5, 10]) #. 하나만 입력하면 y -> x는 0 부터
# plt.show()

# plt.plot((2, 3, 5, 10)) #. 튜플 가능
# plt.show()

# plt.plot(np.array([2,3,5,10])) #. 넘파이 array 가능
# plt.show()

#. 레이블이 있는 데이터 사용하기

data_dict = {'data_x': [1, 2, 3, 4, 5], 'data_y': [2, 3, 5, 10, 8]}

plt.plot('data_x', 'data_y', data=data_dict)
plt.show()