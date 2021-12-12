from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
import numpy as np

#1.데이터
# datasets = load_iris()
datasets = load_boston()

x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=66
)

#2. 모델
# model = DecisionTreeRegressor(max_depth=4)
'''
acc :  0.8774175457631728
[0.03878833 0.         0.         0.         0.00765832 0.29639913
 0.         0.05954596 0.         0.01862509 0.         0.
 0.57898318]
 '''
model = RandomForestRegressor()
'''
acc :  0.929768446800315
[0.03730612 0.00098281 0.00722616 0.00069545 0.02124385 0.38704305
 0.01543431 0.06364649 0.00390724 0.01418505 0.01706015 0.01315657
 0.41811277]
'''


#3. 훈련
model.fit(x_train, y_train)

#4 평가, 예측
acc = model.score(x_test, y_test)
print('acc : ', acc)
# acc :  0.9333333333333333

print(model.feature_importances_)

import matplotlib.pyplot as plt

def plot_feature_importance_dataset(model):
      n_features = datasets.data.shape[1]
      plt.barh(np.arange(n_features), model.feature_importances_,
            align='center')
      plt.yticks(np.arange(n_features), datasets.feature_names)
      plt.xlabel("Feature Importances")
      plt.ylabel("Features")
      plt.ylim(-1, n_features)

plot_feature_importance_dataset(model)
plt.show()