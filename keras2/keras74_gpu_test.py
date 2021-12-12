import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus :
    try : 
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        #. 돌릴 gpu설정
        #. Nvida GPU Utillztion 사용하기
    except RuntimeError as e:
        print(e)
        