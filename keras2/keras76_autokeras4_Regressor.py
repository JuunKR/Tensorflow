from sklearn import datasets
from sklearn.datasets import load_boston
import autokeras as ak
import pandas as pd


datasets = load_boston()
x = datasets.data
y = datasets.target

model = ak.StructuredDataRegressor(overwrite=True, max_trials=1)

# ak.StructuredDataClassifier

model.fit(x, y, epochs=2, validation_split=0.2)

results = model.evaluate(x, y)
print(results)

# autokeras.com 에 공식문서가 있음

