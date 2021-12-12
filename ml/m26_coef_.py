
#1.데이터
x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-3, 65, -19, 11, 3, 47, -1, -7, -47, -25]

# import matplotlib.pyplot as plt

# plt.plot(x, y)
# plt.show()

import pandas as pd
df = pd.DataFrame({'X' : x , 'Y': y})
print(df)
print('구분선 ############################################')
print(df.shape)
print('구분선 ############################################')
x_train = df.loc[:, 'X']
y_train = df.loc[:, 'Y']
print(x_train.shape, y_train.shape)
x_train = x_train.values.reshape(len(x_train), 1) # (10, ) -> (10, 1) .values = 넘파이로 바뀜
print(x_train.shape, y_train.shape)
print(type(x_train))

#2. 모델
from sklearn.linear_model import LinearRegression
model = LinearRegression()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_train, y_train)
print('score : ', score)

print('기울기 : ', model.coef_)  #. cofficient 계수
print('절편 : ', model.intercept_)

'''
기울기 :  [2.]
절편 :  3.0
'''