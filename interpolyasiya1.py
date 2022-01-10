#.3 interpoliasiya

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
x=np.arange(-10,18,0.5)
y=12*x + 3*x**2-1
function=interpolate.interp1d(x,y,kind="quadratic")

x_new=np.arange(-10,18,0.5)
y_new=np.array([])
for i in range(len(x_new)):
    y_new=np.append(y_new,function(x_new[i]))

plt.plot(x_new,y_new,"o")
plt.xlabel("X oxu ")
plt.ylabel("Y oxu ")
plt.show()
