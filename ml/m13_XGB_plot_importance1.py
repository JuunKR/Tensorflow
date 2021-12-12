import numpy as np
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier #부스팅 계열은 확장판 같은 느낌 / 트리 구조의 확장
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



#1.데이터
datasets = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=66
)

#2. 모델
# model = DecisionTreeClassifier(max_depth=4)
# [0.         0.0125026  0.03213177 0.95536562]  #. 다 더하면 1 // 네개의 컬럼이 훈련에 대한 영향도. / 상대적임 절대적이지 않음 . /  이 데이터에 한해 디시젼트리를 적용했을때 쓰레기라는 뜻임 -> train_size 바꿔보자
# model = RandomForestClassifier()
# acc :  0.9666666666666667
# [0.11078963 0.0252825  0.41819652 0.44573135]

#. model = GradientBoostingClassifier()
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4 평가, 예측
acc = model.score(x_test, y_test)
print('acc : ', acc)
# acc :  0.9333333333333333

print(model.feature_importances_)


import matplotlib.pyplot as plt
import numpy as np


# def plot_feature_importance_datasets(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, 
#              align = "center")
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)

# plot_feature_importance_datasets(model)
# plt.show()

plot_importance(model)
plt.show()
