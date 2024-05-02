import numpy as np 
import math 
 
n = 100
dt = 0.01
D=.785
t_list = np.zeros(1000)
for i in range(len(t_list)):
    t_list[i] = i * dt
for i in range(n):
    x = 0
    y = 0 
    z = 0
    with open('data/data_{}.txt'.format(i), 'w') as f:
        for t in t_list:
            x = x + math.sqrt(2*D*dt)*np.random.normal(0,1)
            y = y + math.sqrt(2*D*dt)*np.random.normal(0,1)
            z = z + math.sqrt(2*D*dt)*np.random.normal(0,1)
            f.write("{} {} {} {} {} \n".format(t,x,y,z,x**2 + y**2 +z**2))
