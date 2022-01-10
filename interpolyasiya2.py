import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

x=np.arange(0,105,5)
y=3*x**2 -9*x

a= np.interp(7.7,x,y)
function=interpolate.interp1d(x,y,kind="quadratic")
plt.plot(x,y,"o")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
print(a)
