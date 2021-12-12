from http.client import RESET_CONTENT
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255

#2. 모델
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                    loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batches = [1000, 2000, 3000, 4000, 5000] #, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.3, 0.4, 0.5]
    return {'batch_size': batches, "optimizer": optimizers, 'drop': dropout} 

hyperparameters = create_hyperparameter()   
# print(hyperparameters)

# model2 = build_model()


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
model2 = KerasClassifier(build_fn=build_model, verbose=1,) #validation_split=0.2) # epochs=2) # 여기도 먹힘

model = GridSearchCV(model2, hyperparameters, cv=2) # cv는 크로스 발리데이션 우리는 이전에 kfold를 썻음 // 숫자도 가능 
# 사이킥런하고 머신러닝은 여기에 넣는게 가능했다 하지만 텐서플로 모델은 불가능하다. / 즉 텐서플로 모델을 사이킥런 형식으로 바꾼다. = 사이킥런으로 감싼다 wrapping9

model.fit(x_train, y_train, verbose=1, epochs=2,) #validation_split=0.2) # 여기도 먹힘

print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)
acc = model.score(x_test, y_test)
print("최종 스코어 : ", acc)







