from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten #이미지는 2D 1D는 선 3D는 입체


model = Sequential()                                            #(N, 5, 5, 1)
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(5, 5, 1))) #(N, 4, 4, 10) N = batch_size
model.add(Conv2D(20, (2,2), activation='relu'))                 #(N, 3, 3, 20)
model.add(Flatten())                                            #(N, 180)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.summary()

#과제 1. Conv2D의 디폴트 엑티베이션
#과제 2. Conv2D summary의 파라미터 갯수 완벽 이해 
