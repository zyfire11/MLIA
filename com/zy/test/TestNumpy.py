import numpy

data = [1, 2, 3, 4, 5, 6]
x = numpy.array(data)
print(x)
print(x.dtype)

data = [[1, 2], [3, 4], [5, 6]]
x = numpy.array(data)
print(x)
print(x.ndim)
print(x.shape)

x = numpy.zeros(6)
print(x)
x = numpy.zeros_like(data)
print(x)
x = numpy.zeros((2, 3))
print(x)
x = numpy.ones((2, 3))
print(x)
x = numpy.empty((3, 3))
print(x)

x = numpy.arange(0, 7, 2, float)
print(x)

x = numpy.identity(3)
print(x)
x = numpy.eye(3, 3, 0)
print(x)

x = numpy.array([[[1, 2], [3,4]], [[5, 6], [7,8]]])
print(x)

x = numpy.arange(0, 6)
x = x.reshape((2, 3))
print(x)
print(x.T)

print(numpy.dot(x, x.T))
