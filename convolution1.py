#4. Convolution

import numpy as np

v=np.array([0,1,-2,1,2,3,0,1])
t=np.array([1,0,-1])#(-1,0,1)

result = np.convolve(v,t,mode="same")
print(result)
