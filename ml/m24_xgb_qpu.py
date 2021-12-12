from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = load_boston()
x = datasets['data']
y = datasets['target']

print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.8, shuffle=True, random_state=66) 

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


#2. 모델
model = XGBRegressor(n_estimators=2000, learning_rate=0.05, #n_jobs=7, 
                        tree_method='gpu_hist'
) #njobs 코어수, gpu로 돌리기


#3. 훈련

import time
start_time = time.time()
model.fit(x_train, y_train, verbose=1, eval_metric='rmse',  #. 'mae','logloss'], 
            eval_set=[(x_train, y_train), (x_test, y_test)],
            gpu_id=0, #. gpu 가 여러장 이상일 때 
            predictor='gpu_predictor', #. cpu_predictor 이 옵션은 본인이 알아보기
            
)

print('걸린시간 : ', time.time() - start_time)

'''
njobs= 1 
걸린시간 :  2.2764365673065186

njobs= 2
걸린시간 :  1.8208491802215576

n_jobs=3
걸린시간 :  1.7907347679138184

n_jobs=4
걸린시간 :  1.7295644283294678

n_jobs=5
걸린시간 :  1.7023825645446777

n_jobs=6
걸린시간 :  1.7659573554992676

n_jobs=7
걸린시간 :  1.725717544555664

njobs= 16
걸린시간 :  2.2065467834472656

gpu
걸린시간 :  10.291249752044678

'''



