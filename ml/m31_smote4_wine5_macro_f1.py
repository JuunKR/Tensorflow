# 번외 젯슨나노? -> 라즈베리 파이 ? -> 로봇?  => 우분투를깔 수 있음 

#. 수치데이터의 증폭 
# f1에서는 상승가능성 높음 acc에서는 떨어질 가능성 있음 스모트는 케이네이버를 쓴다


#실습 라벨의 범위를 바꿀것

#기존
#3,4, - 0
#5,6,7, - 1
#8,9 - 2

#범위 변경
#3,4,5 - 0
#6 - 0
#7,8,9 - 2

# 스모그 하기전 후 비교해볼것
# 더좋아지는 기준은 macro-f1



from imblearn.over_sampling import SMOTE
from numpy.lib.function_base import average
from pandas.core.algorithms import value_counts
from scipy.sparse import data
from sklearn import datasets
from sklearn.datasets import load_wine
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
import warnings

from xgboost.training import train

from sklearn.metrics import accuracy_score, f1_score # f1은 이진에서만 사용 다중에서 사용하려면 매크로 사용

warnings.filterwarnings('ignore')

datasets = pd.read_csv('..\\_data\\winequality-white.csv', index_col=None, header=0, sep=';')

datasets = datasets.values
x = datasets[:, :11] # (4898, 11) 
y = datasets[:, 11] # (4898,)
print(x.shape, y.shape) 

print(pd.Series(y).value_counts())
'''
6.0    2198
5.0    1457
7.0     880
8.0     175
4.0     163
3.0      20
9.0       5
'''

###########################################################
### 라벨 대통합 !!! 9를 8로 바꿔라
###########################################################
print('=========================================================')

for i,j in enumerate(y):
    if j == 9:
        y[i] = 2
    elif j == 8:
        y[i] = 2
    elif j == 7:
        y[i] = 1
    elif j == 6:
        y[i] = 1
    elif j == 5:
        y[i] = 1
    elif j == 4:
        y[i] = 0
    elif j == 3:
        y[i] = 0



print(pd.Series(y).value_counts())



x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66)#, stratify=y )#. default = None y의 라벨의 비율과 동일하게 나옴


print(pd.Series(y_train).value_counts())
'''
6.0    1758
5.0    1166
7.0     704
8.0     140
4.0     130
3.0      16
9.0       4
'''

model = XGBClassifier(n_jobs=-1)

model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)

print('model_score : ', score)

y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
print('f1_socre : ', f1)

# model_score :  0.643265306122449

###################################################### smote 적용 ############################################################
import time

st_time = time.time()

smote = SMOTE(random_state=66) #, k_neighbors=66)
ed_time = time.time() - st_time
print('걸린시간: ',  ed_time)


x_smote_train, y_smote_train = smote.fit_resample(x_train,y_train)

# print(pd.Series(y_smote_train).value_counts())

'''
0    53
1    53
2    53
'''

# print(x_smote_train.shape, y_smote_train.shape) # (159, 13) (159,)

print('smote 전 : ', x_train.shape, y_train.shape)
print('smote 후 : ', x_smote_train.shape, y_smote_train.shape)
print('smote 전 레이블 값 분포 : \n', pd.Series(y_train).value_counts())
print('smote 후 레이블 값 분포 : \n', pd.Series(y_smote_train).value_counts())


model2 = XGBClassifier(n_jobs=-1)

model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score = model2.score(x_test, y_test)

y_pred = model2.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')

print('model_score : ', score)
print('f1_socre : ', f1)



# model_score :  0.6236734693877551

'''
깂이 작아서 나오는 에러 
ValueError: Expected n_neighbors <= n_samples,  but n_samples = 4, n_neighbors = 6

n_neighbors 디폴트는 5이다

연결을 5개까지 가능

네이버 줄이면 연산이 줄기 때문에 성능이떨어진다. 

'''