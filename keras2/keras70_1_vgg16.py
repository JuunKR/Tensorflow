from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16 , VGG19

model = VGG16(weights= 'imagenet', include_top=False, # 이거는 내가 개인설정을 할 수 있게하는 셋팅 / 투르는 고정된 값 사용 / false 는 인풋 조정 가능 아웃풋 라인 삭제 = 본인이 만들기
            input_shape=(32,32,3))
# model = VGG16()
# # model = VGG19()
model.trainable = False  # 훈련하지 않는다 = weight의 갱신이 없다. 

model.summary()

print(len(model.weights)) # 26 13의 레이어 13의 바이아스
print(len(model.trainable_weights))
'''
Total params: 14,714,688
Trainable params: 0
Non-trainable params: 14,714,688 ------> 이 전이학습 모델을 훈련시키지 않겠다. // 위의 weights= 'imagenet' weights 값을 그대로 쓰겠다.  
'''
'''
생략한 것임
input_2 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
_________________________________________________________________
'''''''
'''''''
'''''''
flatten (Flatten)            (None, 25088)             0
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544   fc = fully connected 
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000
=================================================================
Total params: 143,667,240
'''

'''
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
'''