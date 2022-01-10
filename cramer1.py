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
