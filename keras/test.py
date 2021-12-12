# import numpy as np

# a = np.arange(1, 10).reshape(3, 3)
# print(a)
# print(np.transpose(a))

# print("Accuracy = 98.93%")

import numpy as np

x_data = np.array(range(1, 101)) # 연속된 열개의 데이터
x_predict = np.array(range(96, 106))


size = 5
96
print(len(x_data)- size +1)

test = x_data[95 : (95+size)]
print(test)


'''
[  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36
  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54
  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72
  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90
  91  92  93  94  95  96  97  98  99 100]
'''

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):  ##
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)
    # 함수를 통과해서 나온 값


dataset = split_x(x_data, size)
print(dataset)