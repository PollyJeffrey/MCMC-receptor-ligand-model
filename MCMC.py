import numpy as np
from scipy import integrate
import scipy.stats

kr = 0.02
nA = 10
nB = 20

observation = np.reshape(np.loadtxt('Synthetic_data.txt'),(10,1))

init_val = np.random.normal(-3, 1)
kf = 10**init_val

def dC_dt(C,t=0):
   return np.array([kf*(nA-C[0])*(nB-C[0])-kr*C[0]])

t = np.linspace(0.0, 100, 100)

C0 = 0
C, infodict = integrate.odeint(dC_dt, C0, t, full_output = 1)
C1 = C[0::10]
#likelihood = np.sum(pow(C1-observation,2))
likelihood = ((1/2*np.pi)**5)*np.exp(-np.sum(pow(C1-observation,2)))

likes = [likelihood]
vals = [init_val]

for i in range(10000):
    numer_old = likes[-1]*scipy.stats.norm(-3, 1).pdf(vals[-1])
    
    val_new = np.random.normal(vals[-1], 1)
    kf = 10**val_new
    C, infodict = integrate.odeint(dC_dt, C0, t, full_output = 1)
    C1 = C[0::10]
    
    #likelihood = np.sum(pow(C1-observation,2))
    likelihood = ((1/2*np.pi)**5)*np.exp(-np.sum(pow(C1-observation,2)))
    prior_density = scipy.stats.norm(-3, 1).pdf(val_new)
    numer_new = likelihood*prior_density

    ratio = numer_new/numer_old
    alpha = min(1,ratio)
    
    if alpha == 1:
        likes.append(likelihood)
        vals.append(val_new)
    else:
        urv = np.random.uniform(0,1)
        if urv < alpha:
            likes.append(likelihood)
            vals.append(val_new)            

import matplotlib.pyplot as plt

#Plot sequence of values for kf
f1 = plt.figure()

plt.plot(vals)

plt.show()

#Plot histogram after burn out phase
kf_posterior = vals[-100:]
true_val = np.log10(0.004)
kf_prior = np.random.normal(-3,1,100)

f2 = plt.figure()

plt.hist(kf_prior,color='C1',alpha=0.5)
plt.hist(kf_posterior,color='C0',alpha=1)
plt.vlines(true_val,0,30,color='red',linestyle='--')
plt.ylim(0,30)

plt.show()
