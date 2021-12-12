import matplotlib.pyplot as plt
import numpy as np

#. 포맷 문자열 사용

# plt.plot([1, 2, 3, 4], [2.0, 3.0, 5.0, 10.0], 'r')
# plt.plot([1, 2, 3, 4], [2.0, 2.8, 4.3, 6.5], 'g')
# plt.plot([1, 2, 3, 4], [2.0, 2.5, 3.3, 4.5], 'b')
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')

# plt.show()

#. color 키워드 사용

# plt.plot([1, 2, 3, 4], [2.0, 3.0, 5.0, 10.0], color='limegreen')
# plt.plot([1, 2, 3, 4], [2.0, 2.8, 4.3, 6.5], color='violet')
# plt.plot([1, 2, 3, 4], [2.0, 2.5, 3.3, 4.5], color='dodgerblue')
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')

# plt.show()

#. Hex code 사용

plt.plot([1, 2, 3, 4], [2, 3, 5, 10], color='#e35f62',
         marker='o', linestyle='--')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

plt.show()