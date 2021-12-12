import numpy as np
from numpy.core.fromnumeric import reshape


aaa = np.array([[  1,   2,   10000,3,   4,   6,  7,   8,   90,  100,   5000], 
                  [1000,2000,3,    4000,5000,6000,7000,8,   9000,10000, 1001]])

# (2,10) -> (10, 2 )
aaa = aaa.transpose()
'''
[[    1  1000]
 [    2  2000]
 [10000     3]
 [    3  4000]
 [    4  5000]
 [    6  6000]
 [    7  7000]
 [    8     8]
 [   90  9000]
 [  100 10000]
 [ 5000  1001]]
'''

# print(aaa)

# print(aaa.shape)

# print(aaa[:, 0])


def outlier(data_out):
    result = []
    for i in range(data_out.shape[1]):
        quartile_1, q2, quartile_3 = np.percentile(data_out[:, i], [25, 50, 75])
        print("Q1 : ", quartile_1)
        print("Q2 : ", q2)
        print("Q3 : ", quartile_3)
        iqr = quartile_3 - quartile_1
        print("IQR : ", iqr)
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)

        result.append(np.where((data_out[:, i]>upper_bound) | (data_out[:, i]<lower_bound)))
        # n = np.count_nonzero((data_out[:, i]>upper_bound) | (data_out[:, i]<lower_bound))
        # result.append([i+1,'columns', m, 'outlier_num :', n])

    return result#  np.array(result)

outliers_loc = outlier(aaa)
print('이상치의 위치 : ', outliers_loc)





import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()