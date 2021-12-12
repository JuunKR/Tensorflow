import tensorflow as tf
from tensorflow.python.distribute.distribute_lib import Strategy

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus :
    try : 
        tf.config.experimental.set_visible_devices([gpus[0],gpus[1]], 'GPU')
        #. 돌릴 gpu설정
        #. Nvida GPU Utillztion 사용하기
    except RuntimeError as e:
        print(e)
        

#. 모델 앞에 놓기
import tensorflow as tf

#. 분산처리; 분산형 학습
# strategy = tf.distribute.MirroredStrategy() #. 오류 확인하기
# strategy = tf.distribute.MirroredStrategy(cross_device_ops= \
#     # tf.distribute.HierarchicalCopyAllReduce()
#     tf.distribute.ReductionToOneDevice()
# )

# strategy = tf.distribute.MirroredStrategy(
#     # devices=['/gpu:0']
#     devices=['/cpu', '/pgu0'] # gpu를 반만씀
#     # 디비아스 옵션에서는 지피유 두개 못넣음
# )

# stratepy = tf.distribute.experimental.CentralStorageStrategy()

# stratepy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
stratepy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    tf.distribute.experimental.CollectiveCommunication.RING
    # tf.distribute.experimental.CollectiveCommunication.NCCL
    # tf.distribute.experimental.CollectiveCommunication.AUTO
)

#. 성능, 속도 거의 비슷 골라서 쓰기


from tensorflow.keras.layers import model
from tensorflow.keras.models import Sequential

with strategy.scope():
    model = Sequential()

    model.compile()

#. 분산할때는 배치가 클수록 좋음
#. NccAll 어쩌구는 gpu오류

#. mnist라고 가정
#. 60000개 0.2 -> 48000개 배치 128 => 375


#.tensorflow.org 튜토리얼과 가이드 각각 분산학습내용이 있음 꼭 보기


#. 속도는 거의 비슷 한개나 두개일때 but 메모리가 늘어나서 덜터짐
#. 배치 크게 주는게 효율적임