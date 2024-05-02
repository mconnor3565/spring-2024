import numpy as np

n = 10
D_list = np.zeros(n)

for i in range(0,n):
    slope,D = np.loadtxt('runs/run{}/slope.txt'.format(i+1), unpack = True, usecols = (0,1))

    D_list[i]=D

for D in D_list:
    print(D)

print("Average D ={:.3f}".format(np.mean(D_list)))