import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.core import Dropout


(x_train, y_train), (x_test, y_test)  = mnist.load_data()
# _는 변수로 받지 않겠다는 뜻  

print(x_train.shape, x_test.shape)
# (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
print(x.shape) # (70000, 28, 28)

x = x.reshape(70000, 28*28)

# 실습
# pca를 통해 0.95이상인 n_components 가 몇개 ?

pca = PCA(n_components=154)
x = pca.fit_transform(x)

# print(x)
# print(x.shape) # (70000, 784)

# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)

# print(sum(pca_EVR))

# cumsum = np.cumsum(pca_EVR)
# print(cumsum)


# print(np.argmax(cumsum >= 0.95)+1)  # 154

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show

x_train = x[:60000, : ]
x_test = x[60000: , :]

print(x_train.shape)
print(x_test.shape)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, ReLU, Dropout


input1 = Input(shape=(154,))
xx = Dense(128, activation='relu')(input1)
xx = Dropout(0.4)(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dense(32, activation='relu')(xx)
xx = Dense(16, activation='relu')(xx)
xx = Dense(16, activation='relu')(xx)
xx = BatchNormalization()(xx)
xx = ReLU()(xx)

output1 = Dense(10, 'softmax')(xx)

model = Model(inputs=input1, outputs=output1)

from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.1) 
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=3) # 

model.fit(x_train,y_train, epochs=1000, batch_size=32, validation_split=0.1, callbacks=[es])

loss = model.evaluate(x_test,y_test) 
print('loss : ', loss[0])
print('acc : ', loss[1])

# loss :  0.13889051973819733
# acc :  0.9771000146865845