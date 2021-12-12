import matplotlib.pyplot as plt
import numpy as np


#. 기본 사용

# plt.plot([1, 2, 3], [4, 4, 4], '-', color='C0', label='Solid')
# plt.plot([1, 2, 3], [3, 3, 3], '--', color='C0', label='Dashed')
# plt.plot([1, 2, 3], [2, 2, 2], ':', color='C0', label='Dotted')
# plt.plot([1, 2, 3], [1, 1, 1], '-.', color='C0', label='Dash-dot')
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# plt.axis([0.8, 3.2, 0.5, 5.0])
# plt.legend(loc='upper right', ncol=4)
# plt.show()

#. line style

# plt.plot([1, 2, 3], [4, 4, 4], linestyle='solid', color='C0', label="'solid'")
# plt.plot([1, 2, 3], [3, 3, 3], linestyle='dashed', color='C0', label="'dashed'")
# plt.plot([1, 2, 3], [2, 2, 2], linestyle='dotted', color='C0', label="'dotted'")
# plt.plot([1, 2, 3], [1, 1, 1], linestyle='dashdot', color='C0', label="'dashdot'")
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# plt.axis([0.8, 3.2, 0.5, 5.0])
# plt.legend(loc='upper right', ncol=4)
# plt.tight_layout()
# plt.show()

#. 튜플 사용
# plt.plot([1, 2, 3], [4, 4, 4], linestyle=(0, (1, 1)), color='C0', label='(0, (1, 1))')
# plt.plot([1, 2, 3], [3, 3, 3], linestyle=(0, (1, 5)), color='C0', label='(0, (1, 5))')
# plt.plot([1, 2, 3], [2, 2, 2], linestyle=(0, (5, 1)), color='C0', label='(0, (5, 1))')
# plt.plot([1, 2, 3], [1, 1, 1], linestyle=(0, (3, 5, 1, 5)), color='C0', label='(0, (3, 5, 1, 5))')
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# plt.axis([0.8, 3.2, 0.5, 5.0])
# plt.legend(loc='upper right', ncol=2)
# plt.tight_layout()
# plt.show()

#. 선 끝 모양

plt.plot([1, 2, 3], [4, 4, 4], linestyle='solid', linewidth=10,
      solid_capstyle='butt', color='C0', label='solid+butt')
plt.plot([1, 2, 3], [3, 3, 3], linestyle='solid', linewidth=10,
      solid_capstyle='round', color='C0', label='solid+round')

plt.plot([1, 2, 3], [2, 2, 2], linestyle='dashed', linewidth=10,
      dash_capstyle='butt', color='C1', label='dashed+butt')
plt.plot([1, 2, 3], [1, 1, 1], linestyle='dashed', linewidth=10,
      dash_capstyle='round', color='C1', label='dashed+round')


plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.axis([0.8, 3.2, 0.5, 5.0])
plt.legend(loc='upper right', ncol=2)
plt.tight_layout()
plt.show()