import numpy as np
from numpy.core.fromnumeric import reshape


aaa = np.array([[  1,   2,   10000,3,   4,   6,  7,   8,   90,  100,   5000], 
                  [1000,2000,3,    4000,5000,6000,7000,8,   9000,10000, 1001]])

# (2,10) -> (10, 2 )
aaa = aaa.transpose()

print(aaa.shape)


from sklearn.covariance import EllipticEnvelope

outliers = EllipticEnvelope(contamination=.2)
outliers.fit(aaa)

results = outliers.predict(aaa)

print(results)
# [ 1  1 -1  1  1  1  1  1  1  1 -1] -1이 아웃라이어 // 근데 왜 8은 아웃라이어가 아닐까 