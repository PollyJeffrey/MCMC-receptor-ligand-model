import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

kf = 0.004
kr = 0.02

nA = 10
nB = 20

def dC_dt(C,t=0):
   return np.array([kf*(nA-C[0])*(nB-C[0])-kr*C[0]])

t = np.linspace(0.0, 100, 100)

C0 = 0
C, infodict = integrate.odeint(dC_dt, C0, t, full_output = 1)
C1 = C[0::10]

data = []
for i in range(10):
   data.append(C1[i] + np.random.normal(0, 0.1*C1[i]))

tt = np.linspace(0,100,10)

fig = plt.figure(figsize=(7,5),dpi=100)
ax = fig.add_subplot(111)
ax.plot(t, C, label='Model output', color='black')
ax.scatter(tt, data, label='Simulated data')

plt.legend(loc=4)
plt.title('Simulated data from a receptor ligand binding model')
plt.ylabel('Concentration of RL complex')
plt.xlabel('Time (secs)')
plt.show()

data2 = np.asarray(data)

np.savetxt("synthetic_data.txt", data2, fmt='%1.6f', delimiter=' ', newline='\n')
