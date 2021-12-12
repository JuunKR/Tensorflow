# 실습 
# 아웃라이어 확인


import pandas as pd
import numpy as np
from xgboost import XGBClassifier


# 1. 데이터

# datasets = pd.read_csv('C:\\Users\\Juun\\Desktop\\programming\\ai\\_data\\winequality-white.csv')
datasets = pd.read_csv('..\\_data\\winequality-white.csv', index_col=None, header=0, sep=';')

print(datasets.info())
print(datasets.head())
print(datasets.shape) # (4898, 12)
print(datasets.describe())

'''
                    fixed acidity  volatile acidity  citric acid  residual sugar    chlorides  free sulfur dioxide  total sulfur dioxide      density           pH    sulphates      alcohol      quality
count               4898.000000       4898.000000  4898.000000     4898.000000  4898.000000          4898.000000           4898.000000  4898.000000  4898.000000  4898.000000  4898.000000  4898.000000
mean (평균)         6.854788          0.278241     0.334192        6.391415     0.045772            35.308085            138.360657     0.994027     3.188267     0.489847    10.514267     5.877909
std  (표준편차)     0.843868          0.100795     0.121020        5.072058     0.021848            17.007137             42.498065     0.002991     0.151001     0.114126     1.230621     0.885639
min                 3.800000          0.080000     0.000000        0.600000     0.009000             2.000000              9.000000     0.987110     2.720000     0.220000     8.000000     3.000000
25%  (분위수)       6.300000          0.210000     0.270000        1.700000     0.036000            23.000000            108.000000     0.991723     3.090000     0.410000     9.500000     5.000000
50%  (분위수)       6.800000          0.260000     0.320000        5.200000     0.043000            34.000000            134.000000     0.993740     3.180000     0.470000    10.400000     6.000000
75%  (분위수)       7.300000          0.320000     0.390000        9.900000     0.050000            46.000000            167.000000     0.996100     3.280000     0.550000    11.400000     6.000000
max                 14.200000          1.100000     1.660000       65.800000     0.346000           289.000000            440.000000     1.038980     3.820000     1.080000    14.200000     9.000000
'''


import matplotlib.pyplot as plt

count_data = datasets.groupby('quality')['quality'].count()
# count_data.plot()
plt.bar(count_data.index, count_data)
plt.show()

print(count_data)
# 적은 수의 값들이 있기 때문에 3~4 등급으로 나누는게 낫지 않을까?
'''
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
'''