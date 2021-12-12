import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 - 4*x + 6

# 2차함수의 미분 그 자리의 기울기. 기울기 = weight 
# 미분했을때 0이 되는 지점이 제일 낮은 지점
# 미분했을때 0이 되는 지점을 모르기 때문에 랜덤으로 값을 처음에 줌
gradient = lambda x: 2*x - 4
#  미분했을 때 0이 되는 곳 x = 2

x0  = 0.0
MaxIter = 20
learning_rate = 0.25

print('step\tx\tf(x)')
print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0)))

'''
step    x       f(x)
00      0.00000 6.00000
'''

for i in range(MaxIter):
    x1 = x0 - learning_rate * gradient(x0)
    x0 = x1

    print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1, x0, f(x0)))




