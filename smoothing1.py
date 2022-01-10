#5. smoothing

import numpy as np
import matplotlib.pyplot as plt

x= np.arange(-5, 5, 0.1)
randvalues=np.random.randn(len(x))

func=-(x+randvalues)**2
temp4=np.array([1,1,1,1,1])
func4=np.convolve(temp4, func, 'same')/5

plt.plot(x, func4)
plt.plot(x, func)
plt.show()
