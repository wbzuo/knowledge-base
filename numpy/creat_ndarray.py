# 习惯写法
import numpy as np

# base 
a = np.array([1, 2, 3])
print(a) # [1 2 3]


# np.empty

a = np.empty([3, 2], order='C')
print(a)
# [[1.e-323 0.e+000]
#  [0.e+000 0.e+000]
#  [0.e+000 0.e+000]]

# numpy.zeros
a = np.zeros([2, 3], order = 'C')

print(a)
# [[0. 0. 0.]
#  [0. 0. 0.]]

# numpy.ones
a = np.ones([2, 3], order = 'C')
print(a)
# [[1. 1. 1.]
#  [1. 1. 1.]]


a = np.ones([2, 3])

b = np.ones_like(a)
c = np.zeros_like(a)

print(b)
# [[1. 1. 1.]
#  [1. 1. 1.]]
print(c)
# [[0. 0. 0.]
#  [0. 0. 0.]]



# np.arange(start, stop, step)
# [0 1 2 3]
print(np.arange(4))
# [2 4 6 8]
print(np.arange(2, 10, 2))

# np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
