#6. determination

from scipy import signal
import numpy as np

object = np.array([[0,1,3,1,0,-1],
                   [-2,1,1,3,0,-1],
                   [-3,0,0,0,2,1],
                   [0,1,3,1,0,-1],
                   [-2,1,1,3,0,-1],
                   [0,1,3,1,0,-1]])

filter=np.array([[0,1],
                 [-1,3]])



result=signal.convolve2d(object,filter,mode='valid')#same de yazmaq olar

edediorta=np.sum(result)/result.size

print(result)
print(edediorta)
