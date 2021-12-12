import numpy as np
from sklearn.datasets import load_boston, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from xgboost.training import train



'''
decomposition 분해
PCA 주성분 분석 -> 통상적으로 비지도 학습이었음 -> 수업에는 주성분 분석기능을 배제할것임.
->요즘은 피처엔지니어링쪽에서 차원축소(컬럼을 압축)하는데 많이 사용하고 있음

mnist 6만 x 28 x 28 x 1 cnn을 dnn으로 구성했을때 reshape(6만 x 784) 컬럼, 피쳐, 특성, 열 
'''

# 1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442,)


pca = PCA(n_components=10) # 컬럼을 7개로 압축
x = pca.fit_transform(x)

print(x)
print(x.shape) # (442, 7)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
# 축소한 데이터의 중요도 / 높은 값이 앞으로 감
# 중요도가 가장 낮은것부터 축소해감
print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)
print(cumsum)

print(np.argmax(cumsum >= 0.94)+1)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()


"""
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=66
)


# 2. 모델
from xgboost import XGBRegressor
model = XGBRegressor()

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('결과 : ', results)

#pca 결과 :  0.3210924574289413
#xgb 결과 :  0.2380270469346017


"""