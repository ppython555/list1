#1. runge-kutta


import numpy as np
def rk2(x,y,h,xfinal):
    A=np.arange(x,xfinal,h)
    for i in A:
        k1=h*(np.exp((x+i)*y))
        k2=h*(np.exp((x+i + 0.5*h)*(y+ 0.5*(k1 +h))))
        y=y+k2
    return y
b=rk2(0,1,0.01,0.2)
print(b)

#runge-kutta
#x^3e^(-2x)-2y
import numpy as np
def rk2(x,y,h,xfinal):
    A=np.arange(x,xfinal,h)
    for i in A:
        k1=h*((x+i)**3*(np.exp((-2)*(x+i))) - 2*y)
        k2=h*((x+i+0.5*h)**3*(np.exp((-2)*(x+i + 0.5*h)))-(2*(y + 0.5*(k1 +h))))
        y=y+k2
    return y
a=rk2(0,1,0.01,0.2)
print(a)
b=rk2(0,1,0.01,0.4)
print(b)

#2. cramer

import numpy as np
total=np.array([[1,1,3,4],
                [2,1,-1,2],
                [1,2,-3,-3]])
A=total[:, [0,1,2]]
Ax=total[:, [3,1,2]]
Ay=total[:, [0,3,2]]
Az=total[:, [0,1,3]]
x=np.linalg.det(Ax)/np.linalg.det(A)
y=np.linalg.det(Ay)/np.linalg.det(A)
z=np.linalg.det(Az)/np.linalg.det(A)
print(['x: ',x,"y: ",y,"z: ",z])

""" cramer misal 2"""
import numpy as np
total=np.array([[1,2,-3,1,7],
                [1,0,-2,2,10],
                [2,-1,0,3,12],
                [0,1,-3,-2,-11]])
A=total[:, [0,1,2,3]]
Ax=total[:, [4,1,2,3]]
Ay=total[:, [0,4,2,3]]
Az=total[:, [0,1,4,3]]
At=total[:, [0,1,2,4]]
x=np.linalg.det(Ax)/np.linalg.det(A)
y=np.linalg.det(Ay)/np.linalg.det(A)
z=np.linalg.det(Az)/np.linalg.det(A)
t=np.linalg.det(At)/np.linalg.det(A)

print(["x: ",x,"y: ",y,"z: ",z,"t:",t])


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


#interpolyasiya


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

#4. Convolution

import numpy as np

v=np.array([0,1,-2,1,2,3,0,1])
t=np.array([1,0,1])
result = np.convolve(v,t,mode="same")
print(result)

""""""

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

#smoot
import numpy as np
import matplotlib.pyplot as plt
x=np.arange(-5,5,0.1)# x uzunlugundadi
randvalues=np.random.rand(len(x))#random qiymetler
function=-(x+randvalues)**2



template=np.array([1,1])
template=np.array([1,1,1])
template=np.array([1,1,1,1])
template=np.array([1,1,1,1,1])
function1=np.convolve(template,function,mode='same')/2
function2=np.convolve(template,function,mode='same')/3
function3=np.convolve(template,function,mode='same')/4
function4=np.convolve(template,function,mode='same')/5


plt.plot(x,function1)
plt.plot(x,function2)
plt.plot(x,function3)
plt.plot(x,function4)

plt.show()



#6. determination

from scipy import signal
import numpy as np

object = np.array([[0,1,3,1,0,-1],
                   [-2,1,1,3,0,-1],
                   [-3,0,0,0,2,1],
                   [0,1,3,1,0,-1],
                   [-2,1,1,3,0,-1],
                   [0,1,3,1,0,-1]])
filter=np.array([[0,1],[-1,3]])

result=signal.convolve2d(object,filter,mode='same')

edediorta=np.sum(result)/result.size

print(result)
print(edediorta)




#runge-kutta
#x^3e^(-2x)-2y
import numpy as np
def rk2(x,y,h,xfinal):
    A=np.arange(x,xfinal,h)
    for i in A:
        k1=h*((x+i)**3*(np.exp((-2)*(x+i))) - 2*y)
        k2=h*((x+i+0.5*h)**3*(np.exp((-2)*(x+i + 0.5*h)))-(2*(y + 0.5*(k1 +h))))
        y=y+k2
    return y
a=rk2(0,1,0.01,0.2)
print(a)
b=rk2(0,1,0.01,0.4)
print(b)



