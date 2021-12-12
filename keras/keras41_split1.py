import numpy as np

a = np.array(range(1, 11)) # 연속된 열개의 데이터
size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)
    # 함수를 통과해서 나온 값



dataset = split_x(a, size)
print(dataset)

x = dataset[:, :4]
y = dataset[:, 4]

print("x : \n", x )
print("y : ", y )