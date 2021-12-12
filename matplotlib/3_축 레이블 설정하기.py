import matplotlib.pyplot as plt
import numpy as np


#. 기본사용

# plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
# plt.xlabel('X-Label') #. 이름지정
# plt.ylabel('Y-Label') #. 이름지정
# plt.show()


#. 여백지정

# plt.plot([1, 2, 3, 4], [2, 3, 5, 10])
# plt.xlabel('X-Axis', labelpad=15)
# plt.ylabel('Y-Axis', labelpad=20)
# plt.show()

#. 폰트설정

# plt.plot([1, 2, 3, 4], [2, 3, 5, 10])
# plt.xlabel('X-Axis', labelpad=15, fontdict={'family': 'serif', 'color': 'b', 'weight': 'bold', 'size': 14})
# plt.ylabel('Y-Axis', labelpad=20, fontdict={'family': 'fantasy', 'color': 'deeppink', 'weight': 'normal', 'size': 'xx-large'})
# plt.show()

# font1 = {'family': 'serif',
#          'color': 'b',
#          'weight': 'bold',
#          'size': 14
#          }

# font2 = {'family': 'fantasy',
#          'color': 'deeppink',
#          'weight': 'normal',
#          'size': 'xx-large'
#          }

# plt.plot([1, 2, 3, 4], [2, 3, 5, 10])
# plt.xlabel('X-Axis', labelpad=15, fontdict=font1)
# plt.ylabel('Y-Axis', labelpad=20, fontdict=font2)
# plt.show()

#. 위치지정
plt.plot([1, 2, 3, 4], [2, 3, 5, 10])
plt.xlabel('X-Axis', loc='right') #. {‘left’, ‘center’, ‘right’}
plt.ylabel('Y-Axis', loc='top')   #. {‘bottom’, ‘center’, ‘top’}
plt.show()