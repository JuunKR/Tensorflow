# 희귀 데이터를 classifier로 만들었을 경우의 에러 확인

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

# 데이터 전처리
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape) # (506, 13)
print(y.shape) # (506,)

print(datasets.feature_names)
print(datasets.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=66) 

# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train) 
# x_test = scaler.transform(x_test)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.svm import LinearSVC, SVC 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor     
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
import tensorflow as tf
from sklearn.metrics import accuracy_score, r2_score
model = make_pipeline(MinMaxScaler() ,LinearRegression())
# 모델
# model = LinearSVC()
#. ValueError: Unknown label type: 'continuous'
# model = SVC()
#. ValueError: Unknown label type: 'continuous'
# model = KNeighborsClassifier()
#. ValueError: Unknown label type: 'continuous'
# model = KNeighborsRegressor()
#. model_score :  0.8407834418231728
# model = DecisionTreeRegressor()
#. model_score :  0.7194699576276645
# model = RandomForestRegressor()
#. model_score :  0.8847107406342373
model = LinearRegression()
#. model_score :  0.8133700013379184



# 컴파일 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)

results = model.score(x_test, y_test)
print('model_score : ', results)

r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)