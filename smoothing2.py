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
#runda plt.plot(x,function1) ve digerlerini ardicil yaz axirdada
#plt.show() yaz'''

plt.plot(x,function1)
plt.plot(x,function2)
plt.plot(x,function3)
plt.plot(x,function4)

plt.show()
