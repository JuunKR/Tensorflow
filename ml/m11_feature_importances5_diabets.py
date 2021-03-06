from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_diabetes

# 1-1. data
datasets = load_diabetes()

x_train, x_test, y_train, y_test = train_test_split(datasets.data, 
                        datasets.target, train_size=0.7,random_state=66)


# 2. model
model = GradientBoostingRegressor()

# 3. fit
model.fit(x_train, y_train)

# 4. pred
r2 = model.score(x_test, y_test)
print('r2 : ', r2) 

print(model.feature_importances_)

'''
r2 :  0.4075401641529469
[0.05601602 0.01245586 0.3406549  0.08600056 0.04979057 0.05747931
 0.06066375 0.01843002 0.2408341  0.0776749 ]
'''

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

'''
GradientBoostingRegressor
r2 :  0.45956469341690287
[0.04684428 0.01878494 0.30428974 0.10471332 0.02694582 0.05384071
 0.04732711 0.01424676 0.31825102 0.06475628]
'''