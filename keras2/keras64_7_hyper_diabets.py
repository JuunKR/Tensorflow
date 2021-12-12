
from sklearn.utils import validation
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

import numpy as np

datasets = load_diabetes()

x = datasets.data # (506, 13) input_dim = 13
y = datasets.target # (506,) output_dim = 1

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.1, shuffle=True, random_state=12)

x_train = x_train.reshape(x_train.shape[0],10,1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0],10,1).astype('float32')/255.

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta

def build_model(drop=0.5, node=2, activation='relu', learning_rate=0.001):
    inputs = Input(shape=(10,1), name='input')
    x = Conv1D(filters = node, kernel_size=2, activation= activation, 
                padding= 'same', name= 'cnn1')(inputs)
    x = Dropout(drop) (x)
    x = Conv1D(filters = node/2, kernel_size=3, activation= activation, 
                padding= 'same', name= 'cnn2')(x)
    x = Dropout(drop) (x)       
    x = Conv1D(filters = node/4, kernel_size=2, activation= activation, 
                padding= 'same', name= 'cnn3')(x)
    x = MaxPooling1D(2,2) (x)
    x = Dropout(drop) (x)
    x = Flatten() (x)
    x = Dense(node, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1, name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate), metrics=['acc'],
                    loss='mse')
    return model                    

def create_hyperparameter():
    batches = [1000,2000]# ,50]
    # optimizer = [Adam, RMSprop, Adadelta]
    activation = ['selu','relu']
    dropout = [0.3, 0.5]
    node = [64, 128]
    epochs = [1, 2]
    validation_split = [0.1, 0.2]
    learning_rate = [0.001, 0.01]
    return {"batch_size":batches,# "optimizer":optimizer,
            "drop":dropout, "activation":activation,
            "node":node, "epochs":epochs,
            "validation_split":validation_split,
            "learning_rate":learning_rate     
    }

hyperparameter = create_hyperparameter()
# model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor = 'val_loss',patience=10)
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, 
                mode='auto', verbose=1, factor=0.9)

model2 = KerasClassifier(build_fn=build_model, verbose=1, 
                        ) # validation_split=0.2) # epochs=2)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

model = RandomizedSearchCV(model2, hyperparameter, cv=2)

model.fit(x_train, y_train, verbose=1, callbacks = [es, lr])


print('best_params_ :',model.best_params_)
print('best_estimator_ :',model.best_estimator_)
print('best_score_ :',model.best_score_)
acc = model.score(x_train, y_train)
print('final_score :', acc)

'''
best_params_ : {'validation_split': 0.1, 'node': 64, 'learning_rate': 0.01, 'epochs': 1, 'drop': 0.5, 'batch_size': 2000, 'activation': 'selu'}     
best_estimator_ : <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000215CC28B3D0>
best_score_ : 0.9971181750297546
final_score : 0.9971181154251099
'''