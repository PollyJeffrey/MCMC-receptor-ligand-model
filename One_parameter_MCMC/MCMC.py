import numpy as np
from scipy import integrate
import scipy.stats

#Set fixed parameter values
kr = 0.02
lT = 0.1
rT = 0.09

#Define the mathematical model
def dC_dt(C,t=0):
   return np.array([kf*(rT-C[0])*(lT-C[0])-kr*C[0]])

#Read in the synthetic data
observation = np.loadtxt('Synthetic_data.txt').T

#Initiate a value for kf
init_val = np.random.normal(-2, 1) #Prior for kf
kf = 10**init_val

#Set the timecourse for the integration
t = np.linspace(0.0, 100, 101)

#Set the initial condition for the model
C0 = 0
#Integrate the model and take the model output at same time points as the data
C, infodict = integrate.odeint(dC_dt, C0, t, full_output = 1)
C1 = C[10::10]
C1_all_rep = np.tile(C1,3)

#initial guess for the standard deviation of the data 
init_sd = 0.001 

#Compute the log likelihood of the data given the model 
log_likelihood = np.sum(np.log(scipy.stats.norm(C1_all_rep,init_sd).pdf(observation)))

#Initiate lists for the likelihoods and the accepted parameter values
likes = [log_likelihood]
vals = [init_val]
sds = [init_sd]

#A loop to run the metropolis hastings algorithm
count = 0
while count < 500:
    print(count)
    #Numerator of the acceptance ratio
    numer_old = likes[-1] + np.log(scipy.stats.norm(-2, 1).pdf(vals[-1]))
    
    #Generate a new parameter value using the proposal distribution
    val_new = np.random.normal(vals[-1], 0.1)
    kf = 10**val_new
    
    #Integrate the model with the new value of the parameter
    C, infodict = integrate.odeint(dC_dt, C0, t, full_output = 1)
    C1 = C[10::10]
    C1_all_rep = np.tile(C1,3)

    #Generate a new value for the SD of the data
    sd_new = np.random.normal(sds[-1], 0.01)
    
    #Compute the log likelihood of the data given the model
    log_likelihood = np.sum(np.log(scipy.stats.norm(C1_all_rep,sd_new).pdf(observation)))

    #Compute the denominator of the acceptance ratio
    prior_density = scipy.stats.norm(-2, 1).pdf(val_new)
    numer_new = log_likelihood + np.log(prior_density)

    #Choose whether or not to accept the new parameter set
    if numer_new > numer_old:
        likes.append(log_likelihood)
        vals.append(val_new)
        sds.append(sd_new)
        count+=1
    else:
        urv = np.random.uniform(0,1)
        if urv < np.exp(numer_new-numer_old):
            likes.append(log_likelihood)
            vals.append(val_new)            
            sds.append(sd_new)
            count+=1

#Plot the results of the MCMC
import matplotlib.pyplot as plt

#Plot sequence of values for kf and sigma
f1 = plt.subplots(1,2,figsize=(10,4),dpi=120)
plt.subplots_adjust(wspace=0.3)
plt.subplot(1,2,1)
plt.plot(vals)
plt.xlabel('Iteration',fontsize=13)
plt.ylabel('Trace of $k_{f}$',fontsize=13)
plt.subplot(1,2,2)
plt.plot(sds)
plt.xlabel('Iteration',fontsize=13)
plt.ylabel('Trace of $\sigma$',fontsize=13)
plt.savefig('Parameter_trace.png',bbox_inches='tight')

#Plot histograms after burn out phase
post_size = 200
kf_posterior = vals[-post_size:]
kf_true_val = np.log10(0.092)
sigma_posterior = sds[-post_size:]
sigma_true_val = 0.003

f2 = plt.subplots(1,2,figsize=(10,4),dpi=120)
plt.subplot(1,2,1)
plt.hist(kf_posterior,bins=6,color='orange',alpha=1,label=r'$k_{f}$ posterior')
plt.vlines(kf_true_val,0,75,color='black',linestyle='--', label=r'$k_{f}$ true value')
plt.xlim(-1.125,-0.975)
plt.legend(fontsize=13)
plt.ylim(0,75)

plt.subplot(1,2,2)
plt.hist(sigma_posterior,bins=6,color='orange',label=r'$\sigma$ posterior')
plt.vlines(sigma_true_val,0,75,color='black',linestyle='--', label=r'$\sigma$ true value')
plt.xlim(0.002,0.005)
plt.legend(fontsize=13)
plt.ylim(0,75)

plt.savefig('Parameter_posteriors.png',bbox_inches='tight')

#Plot kernel density estimates of the posterior distributions
import seaborn as sns

f2 = plt.subplots(1,2,figsize=(10,4),dpi=120)
plt.subplot(1,2,1)
prior = np.random.normal(-2,1,10**3)
sns.kdeplot(prior, shade=True, color='gray', alpha=0.6, label=r'$k_{f}$ prior')
sns.kdeplot(kf_posterior, shade=True, color='green', alpha=0.6, label=r'$k_{f}$ posterior')
plt.vlines(kf_true_val,0,16.5,color='black',linestyle='--', label=r'$k_{f}$ true value')
plt.legend(fontsize=12,loc=2)
plt.ylabel('Density',fontsize=12)
plt.xlabel('$Log_{10}(param)$',fontsize=12)
plt.xlim(-4,0)
plt.ylim(0,16.5)
      
plt.subplot(1,2,2)
sns.kdeplot(sigma_posterior, shade=True, color='green', alpha=0.6, label=r'$\sigma$ posterior')
plt.vlines(sigma_true_val,0,700,color='black',linestyle='--', label=r'$\sigma$ true value')
plt.legend(fontsize=12)
plt.ylabel('Density',fontsize=12)
plt.xlabel('$param$',fontsize=12)   
plt.ylim(0,700)

plt.savefig('Parameter_KDEs.png', bbox_inches='tight')

#Plot the model predictions using the posterior distributions
sims = []
for j in range(post_size):
    def dC_dt(C,t=0):
       return np.array([kf*(rT-C[0])*(lT-C[0])-kr*C[0]])
    
    kf = 10**kf_posterior[j]
    out = integrate.odeint(dC_dt, C0, t)
    sims.append(out)

all_sims = np.hstack(sims)    
med = np.median(all_sims,axis=1)
upp = np.percentile(all_sims,97.5,axis=1)
low = np.percentile(all_sims,2.5,axis=1)

tt = np.linspace(10,100,10)

fig = plt.figure(figsize=(7,5),dpi=120)
ax = fig.add_subplot(111)
ax.plot(t, med, label='Model median', color='black')
ax.fill_between(t, upp, low, color='grey', alpha=0.3, label='Model 95% CI',zorder=1)
for rep in range(3):
    ax.scatter(tt, observation[:,rep], label='Data repeat ' + str(rep+1))

plt.legend(loc=4,fontsize=13)
plt.title('Model fit to the data',fontsize=13)
plt.ylabel('Concentration of complex (nM)',fontsize=13)
plt.xlabel('Time (secs)',fontsize=13)
plt.savefig('Model_fit.png',bbox_inches='tight')

