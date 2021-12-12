#. 앞뒤가 똑같은 오~토인코더~~
# 인코딩을 했다고 다른게 되는것은 아님 
# ex) 얼굴사진을 집어넣고 기미가 사라져도 같은 사람 단지 기미라는 특성이 사라졌을 뿐
# 약한 특성이 사라진다. 
# 데이터가 부족한경우에 증폭용으로도 활용이 가능하다.

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


#@ 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255


#@ 모델
def make_model():
    input_img = Input(shape=(784,))
    encoded = Dense(64, activation='relu')(input_img)
    # encoded = Dense(1064, activation='relu')(input_img)     #. 증폭시킨경우 ;
    decoded = Dense(784, activation='sigmoid')(encoded)
    # decoded = Dense(784, activation='tanh')(encoded)      #. -1 까지 분산이 됨 
    # decoded = Dense(784, activation='linear')(encoded)    #. 양수 음수 무한대 ; 위아래로 더 분산된다 ; 제한이 없다 ; 값에대해서 명확한 특징까지 다 사라져서 흐려짐
    # decoded = Dense(784, activation='relu')(encoded)      #. relu로 하면 더 결과가 더 구려짐
    autoencoder = Model(input_img, decoded)
    #. w와 b때문에 0(검은색)의 값들에 특징들(분산)이 생김
    # autoencoder.summary()
    autoencoder.compile(optimizer = 'adam', loss='mse')
    # autoencoder.compile(optimizer = 'adam', loss='binary_crossentropy')

    
    return autoencoder

autoencoder = make_model()
autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2)

#@ 평가 예측
decoded_imgs = autoencoder.predict(x_test)


import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()



