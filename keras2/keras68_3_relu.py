import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)


x = np.arange(-5, 5, 0.1)
y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# 과제 
# 마이너스 10만과 1을 동급취급 조금 아쉬어.. 
# 파생모델들 알아보자 
# elu, selu, reaky relu....
# 68_3_2, 3, 4 ....로 만들것 !!

