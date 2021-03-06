# 실습 !
# 피쳐임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여 데이터셋을 재구성 후 
# 각 모델별로 돌려서 결과 도출
# 기존 모델결과와 비교
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_iris

# 1-1. data
datasets = load_iris()

# 1. data
x_data = datasets.data
y_data = datasets.target

print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

import pandas as pd

datadf = pd.DataFrame(x_data, columns=datasets.feature_names)

# print(datadf)

#@ cut data
x_data = datadf[['petal length (cm)', 
                'petal width (cm)']]
print(x_data.head)
exit()

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.25, shuffle=True, random_state=1234)

# 2. model
model = DecisionTreeClassifier()

# 3. fit
model.fit(x_train, y_train)

# 4. pred
acc = model.score(x_test, y_test)
print('acc : ', acc) 

print(model.feature_importances_)

import matplotlib.pyplot as plt

# def plot_feature_importance_dataset(model):
#       n_features = datasets.data.shape[1]
#       plt.barh(np.arange(n_features), model.feature_importances_,
#             align='center')
#       plt.yticks(np.arange(n_features), datasets.feature_names)
#       plt.xlabel("Feature Importances")
#       plt.ylabel("Features")
#       plt.ylim(-1, n_features)

# plot_feature_importance_dataset(model)
# plt.show()

'''
DecisionTreeClassifier
banilar
acc :  0.9736842105263158
data cut
acc :  0.9473684210526315
RandomForestClassifier
banilar
acc :  0.9473684210526315
data cut
acc :  0.9473684210526315
GradientBoostingClassifier
banilar
acc :  0.9473684210526315
data cut
acc :  0.9736842105263158
XGBClassifier
banilar
acc :  0.9736842105263158
data cut
acc :  0.9736842105263158
'''
