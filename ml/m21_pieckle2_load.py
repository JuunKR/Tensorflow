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


# #2. 모델
# model = XGBRegressor(n_estimators=2000, learning_rate=0.05, n_jobs=1)


# #3. 훈련
# model.fit(x_train, y_train, verbose=1, eval_metric='rmse',  #, 'mae','logloss'], 
#             eval_set=[(x_train, y_train), (x_test, y_test)],
#             early_stopping_rounds=10
# ) #. validataion 0 = train 1 = test

# 불러오기
import pickle

model = pickle.load(open('C:/Users/Juun/Desktop/programming/ai/_save/xgb_save/m21_pickle.dat', 'rb'))
print('불러왔다')

#4. 평가, 예측
results = model.score(x_test, y_test)
print("results : ", results)

y_predcit = model.predict(x_test)
r2 = r2_score(y_test, y_predcit)
print('r2', r2)

print("==============================================================================================================")
evals_result = model.evals_result()
print(evals_result) # histroy = model.fit()과 비슷함

# # 저장
# import pickle
# pickle.dump(model, open('C:/Users/Juun/Desktop/programming/ai/_save/xgb_save/m21_pickle.dat', 'wb'))



