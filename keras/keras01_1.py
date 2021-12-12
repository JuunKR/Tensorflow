#
from tensorflow.keras.models import Sequential # 모델에는 두가지가 있음 순차형/함수형 
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf
'''
tensorboard --logdir=./logs/fit/ 
'''
#1. 데이터 전처리(특기)
x = np.array([1,2,3]) # 정제된 데이터 / 스칼라 3개 백터 1개
y = np.array([1,2,3]) # 정제된 데이터


#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1)) # Sequential 모델에 Dense 레이어를 추가 / dim은 차원 = X / 앞에는 y 값에 대한 1차원

# Define the Keras TensorBoard callback.
logdir="logs\\fit\\" + 'test'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
 # 로스를 줄이는 방법 : Mean(평균) Squared(제곱) Error 방식 / 1 과 -1 의 거리 차이 0이 나오면 안되기 때문에 절대값을 구해야함 / optimizer 최적화

model.fit(x, y, epochs=4000, batch_size=1, callbacks=[tensorboard_callback])
 # weight 와 Bias(편차) 가 저장됨
 # 세살짜리에게 점찍으라고 하면 못찍음 / epoch 횟수, batch_size 대치 작업 x와 y의 [1,2,3] 개별적으로 한번씩 실행 / 3으로 하면 한번 실행 but 전체적으로 보면 epoch가 1이기 때문에 1번
 # epochs 너무 많이 하면 과부하 걸리기 때문에 조절해야함 -> 하이퍼 
 # 취미가 파라미터? / 하이퍼 파라미터 튜닝(취미) < 데이터 정제가 더 효율이 좋음[전처리] 특기 데이터 전처리

#4. 평가, 예측
loss = model.evaluate(x,y) # loss는 작을 수록 좋다 그러니까 수시로 눈으로 확인 필요
print('loss : ', loss)

result = model.predict([4]) # 위 모델의 4에 대한 예측 값 반환
print('4의 예측값 : ', result)




