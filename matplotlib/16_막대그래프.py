import matplotlib.pyplot as plt
import numpy as np

#. 기본사용

# x = np.arange(3)
# print(x)
# years = ['2018', '2019', '2020']
# values = [100, 400, 900]

# plt.bar(x, values)
# plt.xticks(x, years)

# plt.show()

#. 색상지정

# x = np.arange(3)
# years = ['2018', '2019', '2020']
# values = [100, 400, 900]

# plt.bar(x, values, color='y')
# # plt.bar(x, values, color='dodgerblue')
# # plt.bar(x, values, color='C2')
# # plt.bar(x, values, color='#e35f62')
# plt.xticks(x, years)

# plt.show()

# #각각 다르게
# x = np.arange(3)
# years = ['2018', '2019', '2020']
# values = [100, 400, 900]
# colors = ['y', 'dodgerblue', 'C2']

# plt.bar(x, values, color=colors)
# plt.xticks(x, years)

# plt.show()

#.막대 폭 지정

# x = np.arange(3)
# years = ['2018', '2019', '2020']
# values = [100, 400, 900]

# plt.bar(x, values, width=0.4)
# # plt.bar(x, values, width=0.6)
# # plt.bar(x, values, width=0.8)
# # plt.bar(x, values, width=1.0)
# plt.xticks(x, years)

# plt.show()

#. 스타일 꾸미기

x = np.arange(3)
years = ['2018', '2019', '2020']
values = [100, 400, 900]

plt.bar(x, values, align='edge', edgecolor='lightgray',
        linewidth=5, tick_label=years)

plt.show()