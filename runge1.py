 #1. runge-kutta


import numpy as np
def rk2(x,y,h,xfinal):
    A=np.arange(x,xfinal,h)
    for i in A:
        k1=h*(np.exp((x+i)*y))
        k2=h*(np.exp((x+i + 0.5*h)*(y+ 0.5*(k1 +h))))
        y=y+k2
    return y
