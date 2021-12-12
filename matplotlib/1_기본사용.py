'''
https://setscholars.net/wp-content/uploads/2019/02/visualise-XgBoost-model-with-learning-curves-in-Python.html
'''

import matplotlib.pyplot as plt
import numpy as np


#. 기본 그래프 그리기 및 스타일 지정하기

# a = [1,2,3,4] #. y
# plt.plot(a, 'r-') #.직선
# # plt.plot(a, 'bo') #. 점
# plt.show()

# plt.plot([1,2,3,4], [1,4,9,16], "ro") #. x - y
# plt.axis([0,6,0,20]) #.xmin, xmax, ymin, ymax
# plt.show()


#. 여러개 그래프 그리기
# 200 간격으로 균일하게 샘플된 시간
t = np.arange(0.,5.,0.2) #. 시작, 미만, 간격
'''
[0.  0.2 0.4 0.6 0.8 1.  1.2 1.4 1.6 1.8 2.  2.2 2.4 2.6 2.8 3.  3.2 3.4
 3.6 3.8 4.  4.2 4.4 4.6 4.8]
'''

# 빨간 대쉬, 파란 사각형, 녹색삼각형
# plt.plot(t, t, 'r--', t,t**2, 'bs', t,t**3, 'g^')
# plt.show()

plt.plot(t,t)
plt.plot(t,t**2)
plt.plot(t,t**3)
plt.show()

