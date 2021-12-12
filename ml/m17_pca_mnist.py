import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA


(x_train, _), (x_test, _)  = mnist.load_data()
# _는 변수로 받지 않겠다는 뜻  

print(x_train.shape, x_test.shape)
# (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
print(x.shape) # (70000, 28, 28)

x = x.reshape(70000, 28*28)

# 실습
# pca를 통해 0.95이상인 n_components 가 몇개 ?

pca = PCA(n_components=28*28)
x = pca.fit_transform(x)

print(x)
print(x.shape) # (70000, 784)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)

print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)
print(cumsum)


print(np.argmax(cumsum >= 0.95)+1)  # 154

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()