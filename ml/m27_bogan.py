# 결측치 처리 방법중 하나입니다.

# [1, np.nan, np.nan, 8, 10]

# 결측치 처리
#1. 행 삭제
#2. 0 넣기 (특정값) - [1,0,0,8,10]
#3. 앞의값            [1,1,1,8,10]
#4. 뒤의값            [1,8,8,8,10]
#5. 중위값            [1,4.5,4.5,8,10]
#6. bogan - 리니어
#7. 모델링 - predict 결측치를 뺀 나머지 데이터의 결과치에  model.predict(결측치) 넣어서 결측치 확인
#8. 부스트 계열은 결측치에 대해 자유(?)롭다. - 부스트계열 특히 트리계열은 결측치 신경 안써두 됨/ 물론 해주면 더좋긴함


from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datestrs = ['08/13/2021', '08/14/2021', '08/15/2021', '08/16/2021', '08/17/2021']
dates = pd.to_datetime(datestrs)


print(dates)
print(type(dates)) # <class 'pandas.core.indexes.datetimes.DatetimeIndex'>

ts = Series([1, np.nan, np.nan, 8, 10], index=dates)
print(ts)
'''
2021-08-13     1.0
2021-08-14     NaN
2021-08-15     NaN
2021-08-16     8.0
2021-08-17    10.0
'''

ts_intp_linear = ts.interpolate()
print(ts_intp_linear)
'''
2021-08-13     1.000000
2021-08-14     3.333333
2021-08-15     5.666667
2021-08-16     8.000000
2021-08-17    10.000000
'''

# 값의 기준은 1과 8과 10을 사용하여 선을 긋는다. 두번째와 세번째의 값을 이선의 어디에 해당하는지 맞춰봄 - linear을 기준으로 