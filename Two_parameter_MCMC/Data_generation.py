import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

kf = 0.092
kr = 0.02

lT = 0.1
rT = 0.09

def dC_dt(C,t=0):
   return np.array([kf*(rT-C[0])*(lT-C[0])-kr*C[0]])

t = np.linspace(0.0, 100, 101)

C0 = 0
C, infodict = integrate.odeint(dC_dt, C0, t, full_output = 1)
C1 = C[10::10]

all_data = []
for rep in range(3):
    data = []
    for i in range(10):
       data.append(C1[i] + np.random.normal(0, 0.003))
    all_data.append(data)

tt = np.linspace(10,100,10)

fig = plt.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.plot(t, C, label='Model output', color='black')
for rep in range(3):
    ax.scatter(tt, all_data[rep], label='Data repeat ' + str(rep+1))

plt.legend(loc=4,fontsize=13)
plt.title('Simulated data from the model',fontsize=13)
plt.ylabel('Concentration of complex (nM)',fontsize=13)
plt.xlabel('Time (secs)',fontsize=13)
plt.savefig('Synthetic_data.png')

data2 = np.squeeze(all_data)
np.savetxt("synthetic_data.txt", data2, fmt='%1.6f', delimiter=' ', newline='\n')
