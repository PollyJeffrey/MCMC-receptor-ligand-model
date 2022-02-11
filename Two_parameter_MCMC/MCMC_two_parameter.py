import numpy as np
from scipy import integrate
import scipy.stats

lT = 0.1
rT = 0.09

def dC_dt(C,t=0):
   return np.array([kf*(rT-C[0])*(lT-C[0])-kr*C[0]])

observation = np.loadtxt('Synthetic_data.txt').T

init_kf = np.random.normal(-2, 1) #Prior for kf
kf = 10**init_kf
init_kr = np.random.normal(-2, 1) #Prior for kr
kr = 10**init_kr

t = np.linspace(0.0, 100, 101)

C0 = 0
C, infodict = integrate.odeint(dC_dt, C0, t, full_output = 1)
C1 = C[10::10]
C1_all_rep = np.tile(C1,3)

init_sd = 0.001 #initial guess for the standard deviation of the data 
log_likelihood = np.sum(np.log(scipy.stats.norm(C1_all_rep,init_sd).pdf(observation)))

likes = [log_likelihood]
kf_vals = [init_kf]
kr_vals = [init_kr]
sds = [init_sd]

count = 0
while count < 500:
    print(count)
    numer_old = likes[-1] + np.log(scipy.stats.norm(-2, 1).pdf(kf_vals[-1])) + np.log(scipy.stats.norm(-2, 1).pdf(kr_vals[-1]))
    
    kf_val_new = np.random.normal(kf_vals[-1], 0.1) #Transition function
    kr_val_new = np.random.normal(kr_vals[-1], 0.1) #Transition function
    
    kf = 10**kf_val_new
    kr = 10**kr_val_new
    C, infodict = integrate.odeint(dC_dt, C0, t, full_output = 1)
    C1 = C[10::10]
    C1_all_rep = np.tile(C1,3)

    sd_new = np.random.normal(sds[-1], 0.01)
    log_likelihood = np.sum(np.log(scipy.stats.norm(C1_all_rep,sd_new).pdf(observation)))

    numer_new = log_likelihood + np.log(scipy.stats.norm(-2, 1).pdf(kf_val_new)) + np.log(scipy.stats.norm(-2, 1).pdf(kr_val_new))

    if numer_new > numer_old:
        likes.append(log_likelihood)
        kf_vals.append(kf_val_new)
        kr_vals.append(kr_val_new)
        sds.append(sd_new)
        count+=1
    else:
        urv = np.random.uniform(0,1)
        if urv < np.exp(numer_new-numer_old):
            likes.append(log_likelihood)
            kf_vals.append(kf_val_new)
            kr_vals.append(kr_val_new)
            sds.append(sd_new)
            count+=1

import matplotlib.pyplot as plt

#Plot sequence of values for kf and sigma
f1 = plt.subplots(1,3,figsize=(12,4))
plt.subplots_adjust(wspace=0.3)
plt.subplot(1,3,1)
plt.plot(kf_vals)
plt.xlabel('Iteration',fontsize=13)
plt.ylabel('Trace of $k_{f}$',fontsize=13)
plt.subplot(1,3,2)
plt.plot(kr_vals)
plt.xlabel('Iteration',fontsize=13)
plt.ylabel('Trace of $k_{r}$',fontsize=13)
plt.subplot(1,3,3)
plt.plot(sds)
plt.xlabel('Iteration',fontsize=13)
plt.ylabel('Trace of $\sigma$',fontsize=13)
plt.savefig('Parameter_trace_2param.png',bbox_inches='tight')

#Plot histograms after burn out phase
post_size = 50
kf_posterior = kf_vals[-post_size:]
kr_posterior = kr_vals[-post_size:]
kf_true_val = np.log10(0.092)
kr_true_val = np.log10(0.02)
sigma_posterior = sds[-post_size:]
sigma_true_val = 0.003

f2 = plt.subplots(1,3,figsize=(12,4))
plt.subplot(1,3,1)
plt.hist(kf_posterior,bins=6,color='orange',alpha=1,label=r'$k_{f}$ posterior')
plt.vlines(kf_true_val,0,18,color='black',linestyle='--', label=r'$k_{f}$ true value')
#plt.xlim(-1.125,-0.975)
plt.legend(fontsize=13)
plt.subplot(1,3,2)
plt.hist(kr_posterior,bins=6,color='orange',alpha=1,label=r'$k_{r}$ posterior')
plt.vlines(kr_true_val,0,18,color='black',linestyle='--', label=r'$k_{r}$ true value')
#plt.xlim(-1.125,-0.975)
plt.legend(fontsize=13)
plt.subplot(1,3,3)
plt.hist(sigma_posterior,bins=6,color='orange',label=r'$\sigma$ posterior')
plt.vlines(sigma_true_val,0,22,color='black',linestyle='--', label=r'$\sigma$ true value')
#plt.xlim(0.002,0.005)
plt.legend(fontsize=13)
plt.savefig('Parameter_posteriors_2param.png',bbox_inches='tight')

#Plot the model predictions using the posterior distributions
sims = []
for j in range(post_size):
    def dC_dt(C,t=0):
       return np.array([kf*(rT-C[0])*(lT-C[0])-kr*C[0]])
    
    kf = 10**kf_posterior[j]
    kr = 10**kr_posterior[j]
    out = integrate.odeint(dC_dt, C0, t)
    sims.append(out)

all_sims = np.hstack(sims)    
med = np.median(all_sims,axis=1)
upp = np.percentile(all_sims,97.5,axis=1)
low = np.percentile(all_sims,2.5,axis=1)

tt = np.linspace(10,100,10)

fig = plt.figure(figsize=(7,5),dpi=100)
ax = fig.add_subplot(111)
ax.plot(t, med, label='Model median', color='black')
ax.fill_between(t, upp, low, color='grey', alpha=0.3, label='Model 95% CI',zorder=1)
for rep in range(3):
    ax.scatter(tt, observation[:,rep], label='Data repeat ' + str(rep+1))

plt.legend(loc=4,fontsize=13)
plt.title('Model fit to the data',fontsize=13)
plt.ylabel('Concentration of complex (nM)',fontsize=13)
plt.xlabel('Time (secs)',fontsize=13)
plt.savefig('Model_fit_2param.png',bbox_inches='tight')

