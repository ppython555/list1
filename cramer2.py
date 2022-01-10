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
