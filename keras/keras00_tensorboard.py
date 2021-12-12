# import
import tensorflow as tf
from datetime import datetime

# 텐서보드 설정
logdir="logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1)

# fit 안에 설정
'''
model.fit(x, y, epochs=4000, batch_size=1, callbacks=[tensorboard_callback])
'''

# 텐서보드 실행
'''
tensorboard --logdir=./logs/fit/ 
'''
# tensorboard --logdir=./_save/_gragh/ 

