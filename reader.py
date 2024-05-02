import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

t_iters = 1000

s2_avg = np.zeros(t_iters)

n = 100
for i in range(n):
    t,s2 = np.loadtxt('data/data_{}.txt'.format(i), unpack = True, usecols = (0,4))

    s2_avg += s2

s2_avg /= n


def f(x,m,c):
    return m*x + c  

popt, pcov = curve_fit(f,t, s2_avg)

a_fit, b_fit = popt
plt.plot(t,s2_avg)

plt.plot(t,f(t,a_fit,b_fit), 'b')
plt.savefig('plots/sd_vs_t.pdf')

D = a_fit/6

with open("slope.txt", 'w') as f:
    f.write("{}\t{}".format(a_fit,D))
   



