# 이상치 구하기
# [-1000, 1, 2, 3, 4, 6, 7, 8, 90, 100, 5000] 데이터 
# 사분위수  - 중위수 : 데이터의 중간 값 6 
# 1사분위 2 처음과 중위수 사이의 값
# 3사분위 90 중위수와 마지막 사이의 값

# 이상치 처리
#1. 삭제
#2. Nan 처리 후 -> bogan // linear
#3 ..................(결측치 처리 방법과 유사함)
#4 scaler -> Rubsorscaler, QuantileTransformer ...등등 수동으로 해보고 비교해보기
#5 모델링 : tree게열 DT, RF, XG, LGBM....



import numpy as np
aaa = np.array([1, 2, -1000, 3, 4, 6, 7, 8, 90, 100, 500])
print(aaa.shape)

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75]) #여기서 np.mean으로 하면 안됨 분위값으로 하기
    print('1사분위 : ', quartile_1)
    print('q2 : ', q2)
    print('3사분위 : ', quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outliers_loc = outliers(aaa)
print('이상치의 위치 : ', outliers_loc)

'''
1사분위 :  2.5
q2 :  6.0
3사분위 :  49.0
이상치의 위치 :  (array([ 2, 10], dtype=int64),)
'''


# 시각화
# 위 데이터를 boxplot으로 그리시오 !!!

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()