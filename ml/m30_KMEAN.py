# 비지도에서 대표적인 clustering - 군집화
# pca도 y가 필요없다
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd


datasets = load_iris()

irisDF = pd.DataFrame(data = datasets.data, columns=datasets.feature_names)
print(irisDF)
print(irisDF.info())
print(irisDF.head())
print(irisDF.describe())
print(irisDF.shape)

kmean = KMeans(n_clusters=3, max_iter=300, random_state=66) # max_iter = epoche / n_clusters 라벨값을 뽑겠다 3 개를 뽑는다.
kmean.fit(irisDF)

result = kmean.labels_

print(result)
print(datasets.target)

irisDF['cluster'] = kmean.labels_ # 클러스터링해서 생성한 y값
irisDF['target'] = datasets.target # 원래 y값

print(datasets.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

iris_result = irisDF.groupby(['target', 'cluster'])['sepal length (cm)'].count()
print(iris_result)


