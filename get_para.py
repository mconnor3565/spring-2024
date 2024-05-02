import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#calculating translational displacement 
t,dx,dy,dz = np.loadtxt("CoM_displacement.dat", unpack=True, usecols=(0,1,2,3))

inital_pos = np.array([dx[0], dy[0], dz[0]])
disp_vector = np.zeros((len(t),3))
with open("data/dis_data.txt", 'w') as f:
    f.write('#t dx dy dz\n')
    for i in range(len(t)):
        new_pos = np.array([dx[i], dy[i], dz[i]])
        d = np.subtract(new_pos,inital_pos)
        disp_vector[i,0]= d[0]
        disp_vector[i,1]= d[1]
        disp_vector[i,2]= d[2]
        f.write("{} {:.4f} {:.4f} {:.4f}\n".format(t[i], disp_vector[i,0], disp_vector[i,1], disp_vector[i,2])) 

#getting  the head vector 
t,e1x, e1y, e1z, e2x, e2y, e2z = np.loadtxt("e2e_data.txt", unpack=True)
f_vector = np.zeros((len(t),3))
with open("data/h_data.txt", 'w') as f:
    f.write('#t hx hy hz\n')
    for i in range(len(t)):
        e_1 = np.array([e1x[i], e1y[i], e1z[i]])
        e_2 = np.array([e2x[i], e2y[i], e2z[i]])
        numerator = e_2 - e_1
        denominator = np.sqrt(np.dot(numerator, numerator))
        # print('d=', denominator)
        h = numerator / denominator
        f_vector[i,0]= h[0]
        f_vector[i,1]= h[1]
        f_vector[i,2]= h[2]
        f.write("{} {:.4f} {:.4f} {:.4f}\n".format(t[i], f_vector[i,0], f_vector[i,1], f_vector[i,2]))

#getting the parallel displacement   
disp_para = np.zeros((len(t),3))
with open('data/disp_para.txt', 'w') as f:
    f.write('#t disp para\n')
    for i in range(len(t)):
        h_vector = np.array([f_vector[i,0], f_vector[i,1], f_vector[i,2]])
        d = np.array([disp_vector[i,0], disp_vector[i,1], disp_vector[i,2]])
        p = np.dot(h_vector, d)
        f.write("{} {} {}\n".format(t[i], p, p**2)) 

#windowing average 
t, p, p2 = np.loadtxt('data/disp_para.txt', unpack=True)

t_max = len(t)
sample_window_fraction = 0.01
sample_window = int(t_max * sample_window_fraction)
print("Total iterations: {}".format(t_max))
print("Sample window: {} iterations".format(sample_window))

p_sq_avg_sampled = np.zeros(sample_window)
t0_iter_list = np.arange(0, t_max-sample_window)
print("Samples taken: {}".format(len(t0_iter_list)))

t_shortened = t[:sample_window]
for t0_i in t0_iter_list:
    init_p = p[t0_i]
    p_sq_avg_sampled+= (p[t0_i:t0_i+sample_window]-init_p)**2

    # print(init_p)

with open("data/window_displacement_para.dat", "w") as file:
    for t_i in range(len(t_shortened)):
        file.write("{}\t{}\n".format(t_shortened[t_i], p_sq_avg_sampled[t_i]))


#getting D parallel 
def f(x, m, c):
    return m*x + c

p_sq_avg_sampled /= len(t0_iter_list)
Dimension = 1
fit_params, fit_cov = curve_fit(f, t_shortened, p_sq_avg_sampled)
print('Fitted parameters: m = {:.4e}, c = {:.4e}'.format(fit_params[0], fit_params[1]))
print('Diffusion coefficient: D = {:.4e}'.format(fit_params[0]/(2*Dimension)))
print('or, approximately D = {:.4f}'.format(fit_params[0]/(2*Dimension)))

D_para = fit_params[0]/(2*Dimension)
fitline = f(t_shortened, *fit_params)

p, D_0, D_rot_0, D_perp, D_par, D_rot = np.loadtxt("D_info.txt", unpack=True) 
plt.figure(tight_layout=True)
plt.plot(t_shortened, p_sq_avg_sampled, 'k-', label='Data')
plt.plot(t_shortened, fitline, 'r--', label='Fit')
plt.xlabel(r'$t/\tau$', fontsize=18)
plt.ylabel(r'$\langle \Delta S_{\parallel}^2 \rangle$', fontsize=18)
plt.title(r"$D_{{\parallel}} = {:.4e}$".format(D_0*fit_params[0]/(2*Dimension)))
plt.legend(fontsize=14)
plt.savefig('D_parallel_c.pdf')

# #getting D total
# t, p_sq = np.loadtxt("window_displacement.dat", unpack=True)

# Dimension= 1

# def f(x, m, c):
#     return m * x + c

# popt, pcov = curve_fit(f, t, p_sq)

# D_measured = popt[0] / (2.0*Dimension)

# p, D_0, D_rot_0, D_perp, D_par, D_rot = np.loadtxt("D_info.txt", unpack=True) 
# D_PA = (D_para*D_0)
# print('D p = {:.4e}'.format(D_PA))

# D_actual = (D_measured*D_0)
# print('D total = {:.4e}'.format(D_actual))
 
# print(' D perp = {:.4e}'.format(D_actual - D_PA))

 

