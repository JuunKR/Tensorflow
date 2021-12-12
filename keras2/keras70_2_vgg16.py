from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16 , VGG19

vgg16 = VGG16(weights= 'imagenet', include_top=False, # 이거는 내가 개인설정을 할 수 있게하는 셋팅 / 투르는 고정된 값 사용 / false 는 인풋 조정 가능 아웃풋 라인 삭제 = 본인이 만들기
            input_shape=(32,32,3))
# model = VGG16()
# # model = VGG19()
# model.trainable = False  # 훈련하지 않는다 = weight의 갱신이 없다.  /  가중치를 동결한다. 훈련을 동결한다.

vgg16.trainable=False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))

# model.trainable = False # 가중치 동결, 훈련을 동결한다. 

model.summary()

print(len(model.weights)) # 26 13의 레이어 13의 바이아스                           26 -> 30
print(len(model.trainable_weights))  # 동결시켜서 기존의 것이 0                  # 0 -> 4

