import matplotlib.pyplot as plt
import numpy as np

#. 기본사용

# np.random.seed(0)

# n = 50
# x = np.random.rand(n)
# y = np.random.rand(n)

# plt.scatter(x, y)
# plt.show()

#. 색상 및 크기 지정

np.random.seed(0)

n = 50
x = np.random.rand(n)
y = np.random.rand(n)
area = (30 * np.random.rand(n))**2
colors = np.random.rand(n)

plt.scatter(x, y, s=area, c=colors)
plt.show()