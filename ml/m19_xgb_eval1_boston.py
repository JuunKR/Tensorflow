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
model = XGBRegressor(n_estimators=20, learning_rate=0.05, n_jobs=1)


#3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric='rmse',  #, 'mae','logloss'], 
            eval_set=[(x_train, y_train), (x_test, y_test)]
) #. validataion 0 = train 1 = test

#4. 평가, 예측
results = model.score(x_test, y_test)
print("results : ", results)

# y_predcit = model.predict(x_test)
# r2 = r2_score(y_test, y_predcit)
# print('r2', r2)

print("==============================================================================================================")
hist = model.evals_result()
print(hist) # histroy = model.fit()과 비슷함 = evals_result

# eval_resut의 그래프 그려라
import matplotlib.pyplot as plt

epochs = len(hist['validation_0']['rmse'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, hist['validation_0']['rmse'], label='Train')
ax.plot(x_axis, hist['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('Rmse')
plt.title('XGBoost RMSE')
plt.show()

# import matplotlib.pyplot as plt

# epochs = len(results['validation_0']['logloss'])
# x_axis = range(0, epochs)

# fig, ax = plt.subplots()
# ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
# ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
# ax.legend()
# plt.ylabel('Log Loss')
# plt.title('XGBoost Log Loss')
# # plt.show()

# fig, ax = plt.subplots()
# ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
# ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
# ax.legend()
# plt.ylabel('Rmse')
# plt.title('XGBoost RMSE')
# plt.show()



