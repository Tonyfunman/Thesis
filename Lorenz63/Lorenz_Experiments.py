"""
Lorenz63 SIR dual timscale particle filter assimilation over x and y

This code contains information about:
    -functions and global variables used
    -experiments run

@author: Antoine Gilliard
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import entropy
import math as m
from tqdm import tqdm
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import sys
import pandas as pd
import time
import os


######################INITIALISING GLOBAL VARIABLES############################

#THESE PARAMS ARE NOT CONSTANTS NECESSARILY, THEY CAN VARY FROM ONE SIMULATION
#TO THE OTHER IF WE DECIDE TO CHANGE THEM, THEY ARE JUST A USEFUL WAY OF NOT
#PASSING A LOT OF ARGUMENTS IN EVERY SUBFUNCTION

#Period of assimilation for x and y
x_assim=100 #considered a block
y_assim=300
assim_table=[x_assim, y_assim] #For convenience
factor=int(y_assim/x_assim) #Factor that multiplies x_assim to y_assim

#Timestep length (for euler integration scheme)
dt=0.01

#Total length of the simulation (in number of timesteps)
T=24000

x_blocks = int(T/x_assim) #Number of proxies on x in the simulation
y_blocks = int(T/y_assim) #Number of proxies on y in the simulation

#Proxies data structures
x_proxy = np.zeros(x_blocks)
y_proxy = np.zeros(y_blocks)
set_up=False #Checks if proxies have already been set up for experiment

#Number of particles
N=100

#Particles data structure
particles = np.zeros((N,T,3))

#Particle reproduction data structure, indicates the number of daughters of each
#particle for the next block of time
daughters = np.zeros((N,x_blocks),dtype=int)

#Particle origin data structure, indicates the mother of each particle in the
#previous block of time
mother = np.zeros((N,x_blocks),dtype=int)

#Normed likelihoods data structure, indicates the normed likelihoods for each
#particles for the x or y block
likelihoods_x = np.zeros((N,x_blocks))
likelihoods_y = np.zeros((N,y_blocks))

#Final likelihood vector before resampling, is reflected by the daughter particles
likelihoods_vec = np.zeros((N,x_blocks))

#Conserves block averages for x
x_averages = np.zeros((N,x_blocks))

#Conserves block averages for y
y_averages = np.zeros((N,y_blocks))

#The reconstructed trajectory data structure initialisation
reconstructed_trajectory=np.zeros((T,3))

#Indicates whether particles were or weren't resampled for the x_assim period
#in question
was_just_resampled = np.zeros(x_blocks, dtype=int)

#Start position of reference simulation
start=np.array([[5.86190789,  -1.44799352,  32.19982122]]) # Random point from within the simulation

#Proxies noise
std=[2,2,-1]

#Noise to the dynamic
dyna_std=std[0]/x_assim

#Boolean on whether we assimilate on y
y_matters=False

#Boolean on whether we backtrack on y
backtrack=False

#Should we replace trajectories to reconstruct
replacing=False

#For conditional resampling, criterion alpha_entropy
alpha_entropy=0.85

#Time_tracking
start_time=time.time()
total_time=0
time_likelihoods=0
time_simulating=0
time_resampling=0

###############################################################################

###############################FUNCTIONS USED##################################

'''
pre:
    pos_t: [x-vector,y-vector,z-vector], (d-vector of size N particles)
    s,r,b: lorenz parameters
    make_noisy: argument whether the simulation should be made moisy or not
    noise_std: standard deviation to the white noise added for noisy dynamic
returns:
    pos_dot: travel with or without white noise added
'''
def lorenz_step(pos, s=10, r=28, b=8/3, make_noisy=False, noise_std=dyna_std):
    
    #Separates the x,y,z position vectors
    x = pos[0,:]
    y = pos[1,:]
    z = pos[2,:]
    
    #Calculates the derivative for each particle
    x_dot = s * ( y - x )
    y_dot = r * x - y - np.multiply( x , z )
    z_dot = np.multiply( x , y ) - b * z
    
    #Integrates explicitly over timestep dt
    pos_dot = np.stack((x_dot, y_dot, z_dot), axis=0)*dt #travel
    
    if not make_noisy: #normal travel returned
        return pos_dot
    
    else: #noisy travel returned
        noise = np.random.normal(0.0,scale=noise_std,size=(3,len(x)))
        return pos_dot + noise
    


'''
pre:
    pos_s: list of starting points for the simulation
    K: number of timesteps simulated
    make_noisy: whether the simulation should be made noisy or not
    noise_std: standard deviation to the white noise for noisy dynamic
returns:
    pos_t: travel over multiple timesteps
'''
def lorenz_K_steps(pos_s=start, K=T, make_noisy=False, noise_std=dyna_std):
    
    #Creating the vectors
    pos_t=np.zeros((3,K,len(pos_s[:,0])))
    
    #Setting the starting value
    pos_t[:,0,:]=pos_s.T
    
    #Looping
    for i in range(K-1):
        
        #get the travel
        pos_dot = lorenz_step(pos_t[:,i,:], make_noisy=make_noisy, noise_std=noise_std)
        
        #update with found travel
        pos_t[:,i+1,:]=pos_t[:,i,:]+pos_dot
        
    return pos_t.T

reference_simulation=lorenz_K_steps()[0] #The reference simulation, called only once


'''
pre:
    start_type:
        -"Around": starts around the true starting value
        -"In Region": starts from uniform distribution in region of space
        -"Within": starts from within the butterfly ("In Region" some timesteps in)
    std_around: if "Around", std for white noise around start
    T_within: number of timesteps in to get into the "butterfly" from region
    x_range, y_range, z_range: starting random region polyhedron
returns:
    Just edits the particles global structure to set the starting points for each
'''
def set_particles_start(start_type="Around", std_around=2, T_within=100, 
                        x_range=[-20,20], y_range=[-20,20], z_range=[0,50]):
    
    global particles
    
    if start_type == "Around":
        #Compute random deviation around the start
        random_deviation = np.random.normal(0.0,scale=std_around,size=(N,3))
        #Tesselate the start for each particle
        start_tesselate = np.tile(start,(N,1))
        #Set the particles
        particles[:,0,:]=start_tesselate+random_deviation
        
        
    elif start_type == "In Region" or start_type=="Within":
        
        #Draws starting x,y,z for particles from random distribution
        random_x=np.random.uniform(x_range[0],x_range[1],size=N)
        random_y=np.random.uniform(y_range[0],y_range[1],size=N)
        random_z=np.random.uniform(z_range[0],z_range[1],size=N)
        
        stacked=np.stack((random_x,random_y,random_z),axis=0)
        
        if start_type== "In Region": #Takes the values of uniform distribution
            particles[:,0,:]=stacked.T
        else: #Takes random values a total of T_within Lorenz steps ahead
            particles[:,0,:]=lorenz_K_steps(pos_s=stacked.T, K=T_within)[:,T_within-1,:]
        
    else:
        exit("Wrong type start specification")
        
'''
pre:
    start: the reference timestep to start the simulation
    K: the number of timesteps over which the simulation is run
    make_noisy: whether or not the simulation should be made noisy
    noise_std: the standard deviation of the noise
returns:
    Just simulates the particles as required using lorenz_K_steps
'''
def simulate_particles(start, K=x_assim, make_noisy=False, noise_std=dyna_std):
    global particles
    particles[:,start:start+K,:]=lorenz_K_steps(particles[:,start,:], K, make_noisy=make_noisy, noise_std=noise_std)

    
'''
pre:
    block: denotes the x_block at which the average is taken
returns:
    Updates averages, may backtrack if backtrack global is True
'''
def update_averages(block):
    
    global particles, backtrack, factor, x_assim
    
    ref=block*x_assim#Start of the block
    x_averages[:,block] = np.mean(particles[:,ref:(ref+x_assim),0],axis=1)
    
    if y_matters:#If we assimilate on y
        
        block_pos=block%factor#x_block position in y_block
        y_block=int(block/factor)#y_block in question
        
        if backtrack:#Incremental on y-averages
            
            old_weight=block_pos #the old average weight it given by the block position
            old_average=y_averages[:,y_block]
            new_average=np.mean(particles[:,ref:(ref+x_assim),1],axis=1)
            #Now let us calculate the incremental average of y
            y_averages[:,y_block] = (old_average * old_weight+ new_average)/(old_weight+1)
            
        elif block_pos==factor-1:#We've reached the end of a y block, no need to backtrack

            y_averages[:,y_block] = np.mean(particles[:,(ref-block_pos*x_assim):(ref+x_assim),1],axis=1)
            
'''
pre:
    block: denotes the x_assim block on which we evaluate our likelihood
returns:
    Fills the likelihood vectors and returns:
    likelihood_normed: the likelihood normed to the sum
'''
def compute_likelihoods(block):
    
    #For x
    expected = x_proxy[block]*np.ones(N)
    likelihood_x = stats.norm.pdf((x_averages[:,block]-expected)/(std[0]))
    likelihood_x = likelihood_x
    likelihoods_x[:,block]=likelihood_x#Keep the trace of likelihoods on x
    
    likelihood_y = np.ones(N)
    
    #For y
    if y_matters and block%factor==factor-1:
        y_block=int(block/factor)#y_block in question
        expected = y_proxy[y_block]*np.ones(N)
        likelihood_y = stats.norm.pdf((y_averages[:,y_block]-expected)/(std[1]))
        likelihoods_y[:,y_block]=likelihood_y#Keep the trace of likelihoods on y
        
    likelihood_combined=np.nan_to_num(likelihood_x*likelihood_y)#Combination of both likelihoods
    prob_sum=sum(likelihood_combined)
    
    if prob_sum==0:
        return np.ones(N)/N
    else:
        return likelihood_combined/prob_sum
    

'''
pre:
    block: denotes the block at the end of which we do our resampling
    likelihoods: denotes the likelihoods associated with each particle
returns:
    Updates particles, ready for next simulation block
'''

def resample(block, likelihoods):
    
    likelihoods=likelihoods/sum(likelihoods)#normalise for good measure
    
    likelihoods_vec[:,block]=likelihoods #Keep information about the likelihoods prior to resampling
    
    decimal_daughters=likelihoods*N
    normal_daughters=np.floor(decimal_daughters).astype(int)#perform first step of residual resampling
    
    t=block*x_assim+x_assim-1#end of block timestep
    
    y_block=int(block/factor)#y_block equivalent
    y_avgs=np.ones(N)#Table to store y avgs
    
    #Perform second step of residual resampling
    to_complete=N-np.sum(normal_daughters)#Get the number of particles unnacounted for in resampling
    residues=decimal_daughters-normal_daughters
    if sum(residues)==0:
        final_daughters=normal_daughters
    else:
        residues=residues/sum(residues)#making residues probabilities sum to 1
        sample=np.random.choice(a=N, size=to_complete, p=residues)
        final_daughters=normal_daughters+np.bincount(sample, minlength=N)
        
    if y_matters and backtrack: #Ensure the backtracking works (even though we still use incremental averages so shouldnt be problem)
        y_avgs = np.repeat(y_averages[:,y_block],final_daughters)
    
    pos=0#just init outside scope
    
    if replacing:#replace further than last x_block
        nb_nnz = np.count_nonzero(was_just_resampled)
        last_resampling = -1
        if not nb_nnz==0:
            last_resampling = np.max(np.nonzero(was_just_resampled))#last resampling block
        s = (last_resampling+1)*x_assim #starting timestep
        particles[:,s:t+1,:]=np.repeat(particles[:,s:t+1,:],final_daughters,axis=0)
        pos=particles[:,t,:]
    else:
        pos=np.repeat(particles[:,t,:],final_daughters,axis=0)
    
    if block<x_blocks-1: #We're not at the end of the assimilation
        #Update mothers
        mother[:,block+1] = np.repeat(np.arange(N),final_daughters)
        
        #New particle starts
        new_starts=pos.T+lorenz_step(pos.T,make_noisy=True)
        particles[:,t+1,:]=new_starts.T
    
    if y_matters and backtrack:
        y_averages[:,y_block]=y_avgs
        
    was_just_resampled[block]=1
    daughters[:,block]=final_daughters #save the daughter particles
    
    
'''
A linear interpolation from y proxies to x proxy timescale, modifies global y_proxy and std associated
'''
def interpolate_y_to_x():
    
    global x_assim, y_assim, assim_table, factor, x_blocks, y_blocks, x_proxy, y_proxy, std
    
    y_assim=x_assim
    assim_table=[x_assim, y_assim]#For convenience
    factor=int(y_assim/x_assim) #Factor that multiplies x_assim to y_assim
    
    old_y_blocks = y_blocks
    y_blocks = int(T/y_assim) #Number of proxies on y in the simulation
    
    #In order to use np.interp(), define virtual x_axis values for y_proxy
    old_x = np.array(list(range(old_y_blocks)+np.ones(old_y_blocks)*0.5))
    new_x = np.array((list(range(y_blocks))+np.ones(y_blocks)*0.5)/(y_blocks/old_y_blocks))
    y_proxy = np.interp(new_x,old_x,y_proxy)
    
    std=[std[0],std[0],-1]
    

'''
pre:
    actual_start: start of the simulation
    random_start: start of the simulation is random (within the attractor), ignores the argument actual_start
    interpolate_y: boolean argument on whether y proxies should be interpolated to x_proxy timescale
returns:
    Fills the y_proxy and x_proxy global vectors (pseudoproxies on reference run)
'''
def set_up_proxies(actual_start=start,random_start=False, interpolate_y=False):
    
    global reference_simulation,x_proxy,y_proxy,set_up
    
    if random_start: #Possibly change the reference simulation if we want random start
        x=np.random.uniform(-20,20)
        y=np.random.uniform(-20,20)
        z=np.random.uniform(0,50)
        start_out=np.array([[x,y,z]])
        actual_start=lorenz_K_steps(pos_s=start_out, K=500)[0,499,:].reshape(1,3)
        reference_simulation=lorenz_K_steps(pos_s=actual_start)[0]
    
    #Make the x_proxies
    x_vec=reference_simulation[:,0]
    x_means=np.mean(x_vec.reshape((x_blocks,x_assim)),axis=1)
    x_proxy=np.random.normal(loc=x_means ,scale=std[0])
    
    #Make the y_proxies
    y_vec=reference_simulation[:,1]
    y_means=np.mean(y_vec.reshape((y_blocks,y_assim)),axis=1)
    y_proxy=np.random.normal(loc=y_means ,scale=std[1])
    
    if interpolate_y:
        interpolate_y_to_x()
    
    set_up=True
    
'''
Used in conditional resampling on whether or not one should resample at this stage
pre:
    likelihoods: the likelihood vector
returns:
    If under the threshold "alpha_entropy" True, if above False
'''
def condition_entropy(likelihoods):
    global alpha_entropy, N
    if entropy(likelihoods, base=N)<alpha_entropy:#Is the entropy too low? not enough dispersal?
        return True
    else:
        return False
    
'''
pre:
    likelihoods: the likelihood vector
returns:
    The gini coefficient associated with this likelihood vector
'''
def gini(likelihoods):
    # Mean absolute difference
    global N
    # Sort x
    x_s=np.sort(likelihoods)
    
    #Difference from x to next x
    x_diff=x_s[1:]-x_s[:N-1]
    
    # Equivalent to mean absolute difference
    ad=sum(np.arange(1,N)*np.flip(np.arange(1,N))*x_diff)
    
    # Relative mean absolute difference
    g = ad/(np.mean(likelihoods)*N**2)
    
    return g

'''
pre:
    likelihood: a dummy variable for the matter
returns:
    True
'''
def dummy_true_condition(likelihoods):
    return True

'''
pre:
    likelihood: a dummy variable for the matter
returns:
    False
'''
def dummy_false_condition(likelihoods):
    return False


'''
pre:
    reset_particles: whether or not to reset global particle related data structures
    reset_proxies: whether or not to reset global proxy related data structures
returns:
    nothing just resets what needs to be
'''
def reset(reset_particles=False, reset_proxies=False):
    
    if reset_particles:#Reset global vairables associated to the particles
        
        global particles, daughters, mother, likelihoods_x, likelihoods_y, likelihoods_vec, x_averages, y_averages, was_just_resampled
        
        particles = np.zeros((N,T,3))
        daughters = np.zeros((N,x_blocks),dtype=int)
        mother = np.zeros((N,x_blocks),dtype=int)
        likelihoods_x = np.zeros((N,x_blocks))
        likelihoods_y = np.zeros((N,y_blocks))
        likelihoods_vec = np.zeros((N,x_blocks))
        x_averages = np.zeros((N,x_blocks))
        y_averages = np.zeros((N,y_blocks))
        was_just_resampled = np.zeros(x_blocks, dtype=int)
        
    if reset_proxies:#Reset global vairables associated to the proxies
        
        global x_proxy, y_proxy, set_up
        
        x_proxy = np.zeros(x_blocks)
        y_proxy = np.zeros(y_blocks)
        set_up=False
        
'''
pre:
    method: The type of assimilation method you want to use
            -"Base"(Without interpolation)
            -"Base_Interpolation"
            -"Cumulative"
            -"Thowback"
            -"Conditional"
    start_type: The start type you want for your particles
            -"Around"- Around reference start
            -"Within"- Within the attractor
    random_start: Whether the proxies come from a random start
    conditional_condition: Which function should be used for conditional resampling
'''
def run_assimilation(method="Base", start_type="Around", random_start=False, conditional_condition=condition_entropy):
    
    global set_up, y_matters, backtrack, condition, time_simulating, time_likelihoods, time_resampling, replacing
    
    if not set_up:#If proxies are not setup
        if method=="Base_Interpolation":#If we use a method using interpolation
            set_up_proxies(random_start=random_start, interpolate_y=True)
        else:
            set_up_proxies(random_start=random_start)
    
    #Reset particles ahead of assimilation
    reset(reset_particles=True)
    set_particles_start(start_type=start_type)
       
    likelihood_vector=np.ones(N)
    
    #The following gets the correct parameter setting for each method
    if method=="Base-Ingnoring" or method=="Base":
        y_matters=False
        backtrack=False
        condition=dummy_true_condition
        
    if method=="Base_Interpolation":
        y_matters=True
        backtrack=False
        condition=dummy_true_condition
    
    if method=="Backtrack":
        y_matters=True
        backtrack=True
        replacing=True #Works with particle history replacement
        condition=dummy_true_condition
    
    if method=="Cumulative":
        y_matters=True
        backtrack=False
        condition=dummy_false_condition
        
    if method=="Conditional":
        y_matters=True
        backtrack=True
        replacing=True #Works with particle history replacement
        condition=conditional_condition
    
    was_just_resampled[-1]=0 #In order to not start particles noisily
    
    for i in range(x_blocks):
        
        s=i*x_assim #starting timestep of block
        
        chrono=time.time() #keeping track of times, for optimisation
        
        if was_just_resampled[i-1]: # There was just a resample in last block, noisy differentiation required
            simulate_particles(s, K=x_assim, make_noisy=True)
            likelihood_vector=np.ones(N)
        else: # There was no resample in last block, noisy differentiation not required
            simulate_particles(s, K=x_assim, make_noisy=False)
        
        time_simulating = time_simulating+time.time()-chrono#keeping track of times, for optimisation
        chrono=time.time()#keeping track of times, for optimisation
        
        update_averages(i)#Update the averages of the block
        likelihood_vector*=compute_likelihoods(i)#Increment likelihoods
        
        if (sum(likelihood_vector)==0):#Avoid warnings etc, if all likelihoods are useless, might as well set weights to 1
            likelihood_vector=np.ones(N)/N
        else:
            likelihood_vector=likelihood_vector/sum(likelihood_vector)#normalise for good measure
        
        time_likelihoods=time_likelihoods+time.time()-chrono#keeping track of times, for optimisation
        chrono=time.time()#keeping track of times, for optimisation
        
        #Propagating particles to the next block with or maybe without resampling
        if (i%factor==factor-1 or condition(likelihood_vector)): #Will resample based on condition and end of y block
            #If not conditional resampling, condition changes to dummy true or dummy false between the methods
            resample(i, likelihood_vector)#Resample the particles based on current likelihood
            time_resampling=time_resampling+time.time()-chrono
            
        else: #Simulate one step for start of next block if no resampling
            t=i*x_assim+x_assim-1#end of block timestep
            new_starts=particles[:,t,:].T+lorenz_step(particles[:,t,:].T,make_noisy=False)
            particles[:,t+1,:]=new_starts.T
 
############################################################ 
###########FUNCTIONS TO EVALUATE THE PERFORMANCE############
############################################################ 
'''
Particle swarm is used used to reconstruct the trajectory
pre:
    method: Argument which determines how to reconstruct this trajectory
        -"Average": takes the reconstruction of the curves by taking average weighted by daughters
        -"Best-Particle": makes the reconstruction by taking the best particle on each segment
returns:
    Edits the globacl variable reconstructed_trajectory
'''
def reconstruct_trajectory(method="Average"):
    global reconstructed_trajectory
    reconstructed_trajectory=np.zeros((T,3))
    if method=="Average":
        if replacing:
            reconstructed_trajectory=np.mean(particles, axis=0)
        else:
            if backtrack==False:
                if y_matters:
                    weights=np.repeat(np.extract(np.mod(np.arange(N*x_blocks).reshape(N,x_blocks),factor)==factor-1,daughters),y_assim*3).reshape(N,T,3)
                    reconstructed_trajectory=np.average(particles, axis=0, weights=weights)
                else:
                    weights=np.repeat(daughters,x_assim*3).reshape(N,T,3)
                    reconstructed_trajectory=np.average(particles, axis=0, weights=weights)
            else:#TODO, is kind of false, but helps for x-y-z visualisation when backtracking method is used
                weights=np.repeat(daughters,x_assim*3).reshape(N,T,3)
                reconstructed_trajectory=np.average(particles, axis=0, weights=weights)
                
    if method=="Best-Particle":
        if y_matters:
            factors=np.extract(np.mod(np.arange(N*x_blocks).reshape(N,x_blocks),factor)==factor-1,daughters).reshape(N,int(x_blocks/factor))
            selection=np.array(factors==np.max(factors, axis=0), dtype=int)
            weights=np.repeat(selection,y_assim*3).reshape(N,T,3)
            reconstructed_trajectory=np.average(particles, axis=0, weights=weights)
        else:
            selection=np.array(daughters==np.max(daughters,axis=0), dtype=int)
            weights=np.repeat(selection,x_assim*3).reshape(N,T,3)
            reconstructed_trajectory=np.average(particles, axis=0, weights=weights)

'''
pre:
    rec: a reconstructed trajectory
returns:
    RMSE: The value of the RMSE, whcich is in this case the euclidian 
'''
def compute_simulation_RMSE(rec):
    return np.sqrt(np.mean(np.sum((rec-reference_simulation)**2 , axis=1)))



'''
pre:
    rec: a reconstructed trajectory
returns:
    corr: The correlation values of this reconstruction on xyz 
'''
def compute_simulation_corr(rec):
    corr_tab=np.zeros(3)
    for i in range(3):
        corr_matrix=np.corrcoef(rec[:,i],reference_simulation[:,i])#calculates correlation matrix between time series
        corr_tab[i]=corr_matrix[0,1]#takes only the correlations of interest
    return corr_tab

'''
pre:
    rec: a reconstructed trajectory
returns:
    corr_delta: The correlation values of this reconstruction on xyz over x_assim and y_assim (2x3 np array)
'''
def compute_simulation_corr_delta(rec):
    corr_tab_delta_x_y=np.zeros((2,3))
    means_over=[x_assim,y_assim]
    for i in range(2):
        delta=means_over[i]
        #Creation of the time series based on averages
        mean_ref=np.mean(np.reshape(reference_simulation,(int(T/delta),delta,3)), axis=1)
        mean_rec=np.mean(np.reshape(rec,(int(T/delta),delta,3)), axis=1)
        for j in range(3):
            corr_matrix=np.corrcoef(mean_rec[:,j],mean_ref[:,j])#calculates correlation matrix between time series
            corr_tab_delta_x_y[i,j]=corr_matrix[0,1]#takes only the correlations of interest
    return corr_tab_delta_x_y
          
'''
pre:
    plot_stuff: argument that specifies whether or not to give the distance plot
    It evaluates global variables (particles and reference simulation). Also
    plots the swarm of particles on x, y and z visualisation
    plot_xyz_method: Gives the xyz assimilation method to indicate on the graph
returns:
    RMSE,Corr,Corr_delta with:
    RMSE: The RMSE of the reconstruction based on weighted mean
    Corr: Correlations table (corr_x,corr_y,corr_z)
    Corr_delta: Correlations table of values over delta_x and over delta_y
'''
def evaluate_reconstruction(plot_stuff=False, plot_xyz_method="Base", get_correlations=False, get_correlations_delta=False):
    reconstruct_trajectory(method="Average")
    if plot_stuff:#For visualisation
        plot_distance_between_curves(reconstructed_trajectory)
        plot_xyz(reconstructed_trajectory, start=0, end=600,plot_particles=True,plot_observations=True, method=plot_xyz_method)
    RMSE=compute_simulation_RMSE(reconstructed_trajectory)
    Corr=compute_simulation_corr(reconstructed_trajectory)
    Corr_delta=compute_simulation_corr_delta(reconstructed_trajectory)
    return RMSE,Corr,Corr_delta


############################################################ 
##################FUNCTIONS TO PLOT STUFF###################
############################################################
    
'''
    start: timestep for start of plot
    end: timestep for end of plot
    plot_particles: whether or not particles should be included
    plot_observations: whether or not to plot observations on x and y
'''
def plot_xyz(reconstructed_trajectory, true=None, start=0, end=600, plot_particles=False, plot_observations=True, method="Base"):
    
    if true==None:
        true=reference_simulation
        
    x=true[start:end,0]
    y=true[start:end,1]
    z=true[start:end,2]
    
    x_r=reconstructed_trajectory[start:end,0]
    y_r=reconstructed_trajectory[start:end,1]
    z_r=reconstructed_trajectory[start:end,2]
    
    fig, axs = plt.subplots(3,figsize=(6,9))
    fig.suptitle('Observations on particle behaviours XYZ, '+ method)
    
    timesteps=np.arange(600)
    if plot_particles:
        for d in range(3):
            for i in range(x_blocks):
                if ((i+1)*x_assim>end):
                    break;
                if was_just_resampled[i]==1:
                    for p in range(N):
                        if daughters[p,i]==0:
                            col='lightgrey'
                            if method=="Cumulative":
                                factor=int(y_assim/x_assim)
                                if (i+1)%factor!=0:
                                    next_y_block=i+(factor-(i+1)%factor)
                                    if daughters[p,next_y_block]!=0:
                                        col='red'
                        else:
                            col='red'
                        axs[d].plot(timesteps[i*x_assim:(i+1)*x_assim],particles[p,i*x_assim:(i+1)*x_assim,d], color=col, lw=0.2)
                else:
                    if i==x_blocks-1:
                        for p in range(N):
                            if daughters[p,i]==0:
                                col='lightgrey'
                                if method=="Cumulative":
                                    factor=int(y_assim/x_assim)
                                    if (i+1)%factor!=0:
                                        next_y_block=i+(factor-(i+1)%factor)
                                        if daughters[p,next_y_block]!=0:
                                            col='red'
                            else:
                                col='red'
                            axs[d].plot(timesteps[i*x_assim:(i+1)*x_assim],particles[p,i*x_assim:(i+1)*x_assim,d], color=col, lw=0.2)
                    else:
                        for p in range(N):
                            if daughters[p,i]==0:
                                col='lightgrey'
                                if method=="Cumulative":
                                    factor=int(y_assim/x_assim)
                                    if (i+1)%factor!=0:
                                        next_y_block=i+(factor-(i+1)%factor)
                                        if daughters[p,next_y_block]!=0:
                                            col='red'
                            else:
                                col='red'
                            axs[d].plot(timesteps[i*x_assim:(i+1)*x_assim+1],particles[p,i*x_assim:(i+1)*x_assim+1,d], color=col, lw=0.2)
                    
                    
    
    axs[0].plot(x,color='blue')
    axs[1].plot(y,color='blue')
    axs[2].plot(z,color='blue')
    
    axs[0].plot(x_r,color='lime')
    axs[1].plot(y_r,color='lime')
    axs[2].plot(z_r,color='lime')
    
    if plot_observations:
        axs[0].scatter(np.arange(x_assim/2,end,x_assim),x_proxy[0:int(end/x_assim)], color='k', zorder=2)
    if method != "Base":
        axs[1].scatter(np.arange(y_assim/2,end,y_assim),y_proxy[0:int(end/y_assim)], color='k', zorder=2)
        
    axs[0].set_ylabel("x value")
    axs[1].set_ylabel("y value")
    axs[2].set_ylabel("z value")
    
    plt.savefig('xyz_for_'+str(x_assim)+'_'+method+'.png')

    plt.show()

'''
pre:
    rec: a reconstructed trajectory
returns:
    A plot of the distance between the curves
'''
def plot_distance_between_curves(rec):
    rmse_step=np.sqrt(np.sum((rec-reference_simulation)**2 , axis=1))

    fig,ax=plt.subplots()
    ax.plot(rmse_step)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Distance")
    ax.set_title("Distance from actual measurements over time")
    plt.show()
    
    
''' 
    Given:
        true: the true value of the simulation
        start, end: the start and end of simulation for which the particles are plotted
        title: title of graph
        put_equilibria: boolean, whether or not it should show lorenz equilibria
        save_fig: boolean, whether or not figure should be saved
    Returns:
        a plot of the behaviour of those particles in that timeframe
'''
def plot_particles(start, end, true=None, title="Particles Plotted", put_equilibria=False, save_fig=False):
    
    if true==None:
        true=reference_simulation
        
    x=true[start:end,0]
    y=true[start:end,1]
    z=true[start:end,2]
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in range(N):
        ax.plot(particles[i,start:end,0], particles[i,start:end,1], particles[i,start:end,2], lw=0.5, color='red')
    ax.plot(x, y, z, lw=0.5, color='blue')
    
    if put_equilibria:
        ax.scatter(0,0,0, color='red')
        ax.scatter(m.sqrt(8/3*(28-1)),m.sqrt(8/3*(28-1)),27, color='red')
        ax.scatter(-m.sqrt(8/3*(28-1)),-m.sqrt(8/3*(28-1)),27, color='red')
        
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)
    
    if save_fig:
        plt.savefig('figs/'+title.replace(" ", "_")+".png")
    plt.show()

    
'''
    Given:
        true: the true value of the simulation
        start, end: the start and end of reconstruction to be plotted
        title: title of graph
    Returns:
        A plot of the reconstructed and true trajectories in that timeframe
'''
def plot_reconstruction(start, end, true=None, title="Reconstructed Trajectory vs True Trajectory"):
    
    if true==None:
        true=reference_simulation
        
    x=true[start:end,0]
    y=true[start:end,1]
    z=true[start:end,2]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    x_r=reconstructed_trajectory[start:end,0]
    y_r=reconstructed_trajectory[start:end,1]
    z_r=reconstructed_trajectory[start:end,2]
    
    ax.plot(x, y, z, lw=0.5, color='blue')
    ax.plot(x_r, y_r, z_r, lw=0.5, color='green')
    
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)

    plt.show()
    

############################################################ 
##############FUNCTIONS TO RUN MULTIPLE TESTS###############
############################################################
    
'''
pre:
    method: The type of assimilation method you want to use
            -"Base"
            -"Base_Interpolation"
            -"Cumulative"
            -"Backtrack"
            -"Conditional"
    start_type: The start type you want for your particles
            -"Around"
            -"Within"
    random_start: Whether the reconstructed simulation comes from a random start
    plot_stuff: argument that specifies whether or not to give the distance plot
    It evaluates global variables (particles and reference simulation). Also
    plots the swarm of particles on x, y and z visualisation
returns:
    the "evaluate_reconstruction" return for that specific run
'''
def test_method(method="Base", start_type="Around",random_start=False, resetting=True, plot_stuff=False):
    if resetting:
        reset(reset_proxies=True)
    run_assimilation(method=method, start_type=start_type, random_start=random_start)
    return evaluate_reconstruction(plot_stuff = plot_stuff)+tuple([average_gini_at_resampling()])+tuple([average_entropy_at_resampling()])

'''
Tests for a varying range of y assimilation timescales
pre:
    This function takes the same arguments as the test_method function, and sets
    resetting to True and plot_stuff to False
    y_prox: The range of y_proxy factors used
returns:
    RMSE_TAB,Corr_TAB,Corr_delta_TAB: The tables of values of evaluation for each run
'''
def test_Y_Range(method="Base", start_type="Around", random_start=False,
                 y_factor=[2,3,4,6,8,10]):
    
    global factor, std, x_assim, y_assim, assim_table, set_up, y_blocks, y_proxy, likelihoods_y, y_averages
    
    nb_y_factoro=len(y_factor)
    if method=="Base": #No need to test for all y values
        y_factor=[2]
    nb_y_factor=len(y_factor)
     
    #Initialises the data tables
    RMSE_TAB=np.zeros(nb_y_factor)
    Corr_TAB=np.zeros((nb_y_factor,3))
    Corr_delta_TAB=np.zeros((nb_y_factor,2,3))
    Gini_TAB=np.zeros(nb_y_factor)
    Entropy_TAB=np.zeros(nb_y_factor)
    
    for i in range(nb_y_factor):
        #Changes y proxy and everything related to it
        factor=y_factor[i]
        y_assim=x_assim*factor
        assim_table=[x_assim, y_assim] #For convenience
        y_blocks = int(T/y_assim) #Number of proxies on y in the simulation
        y_proxy = np.zeros(y_blocks)
        set_up=False #Checks if proxies have already been set up for experiment
        likelihoods_y = np.zeros((N,y_blocks))
        y_averages = np.zeros((N,y_blocks))
        std=[std[0],std[0]/m.sqrt(factor),-1]
        RMSE_TAB[i],Corr_TAB[i,:],Corr_delta_TAB[i,:,:],Gini_TAB[i],Entropy_TAB[i]=test_method(method=method, start_type=start_type, random_start=random_start)
            
    if method=="Base": #Sends the tables duplicated along the y_factor axis (for simplicity)
        #Tiling for same format return
        RMSE_TAB=np.tile(RMSE_TAB,(nb_y_factoro))
        Gini_TAB=np.tile(Gini_TAB,(nb_y_factoro))
        Entropy_TAB=np.tile(Entropy_TAB,(nb_y_factoro))
        Corr_TAB=np.tile(Corr_TAB,(nb_y_factoro,1))
        Corr_delta_TAB=np.tile(Corr_delta_TAB,(nb_y_factoro,1,1))
    
    return RMSE_TAB,Corr_TAB,Corr_delta_TAB,Gini_TAB,Entropy_TAB

'''
pre:
    This function takes the same arguments as the test_Y_Range function
    N_range: The number of particles range
    runs: The number of runs
returns:
    RMSE_TAB: The values for cross RMSE for each run
'''
def cross_trial(method="Base", start_type="Around", random_start=False,
                  N_range=[10,18,30,56,100,180,300,560,1000,1800,3000],
                 y_prox=[2,3,4,6,8,10], runs=10):
    
    global N, y_assim, x_assim, std,x_blocks,y_blocks, factor
    RMSE_TAB=np.zeros((len(N_range),len(y_prox),runs))
    Corr_TAB=np.zeros((len(N_range),len(y_prox),runs,3))
    Corr_delta_TAB=np.zeros((len(N_range),len(y_prox),runs,2,3))
    Gini_TAB=np.zeros((len(N_range),len(y_prox),runs))
    Entropy_TAB=np.zeros((len(N_range),len(y_prox),runs))
    for k in tqdm(range(runs)):
        for i in range(len(N_range)):
            N=N_range[i]
            RMSE_TAB[i,:,k],Corr_TAB[i,:,k,:],Corr_delta_TAB[i,:,k,:,:],Gini_TAB[i,:,k], Entropy_TAB[i,:,k]=test_Y_Range(method=method, start_type=start_type, random_start=random_start,y_factor=y_prox)
    return RMSE_TAB,Corr_TAB,Corr_delta_TAB, Gini_TAB, Entropy_TAB

'''
Measures the average entropy of the likelihoods at resampling
returns:
    the average entropy at resampling
'''
def average_entropy_at_resampling():
    global was_just_resampled, likelihoods_vec, N
    entropy_sum=0
    for i in range(x_blocks):
        if was_just_resampled[i]:
            entropy_sum+=entropy(likelihoods_vec[:,i],base=N)
    return entropy_sum/sum(was_just_resampled)

'''
Measures the average entropy of the likelihoods at resampling
returns:
    the average gini coefficient at resampling
'''
def average_gini_at_resampling():
    global was_just_resampled, likelihoods_vec, N
    entropy_sum=0
    for i in range(x_blocks):
        if was_just_resampled[i]:
            entropy_sum+=gini(likelihoods_vec[:,i])
    return entropy_sum/sum(was_just_resampled)

'''
A summary of the time stats in the simulation(s), just call it to print time stats
'''   
def time_stats():
    global total_time,time_likelihoods,time_simulating,time_resampling
    time_external=total_time-time_likelihoods-time_simulating-time_resampling
    print("Times spent is as follows:")
    print("Total time, ", total_time)
    print("Time resampling", time_resampling, " meaning ", time_resampling/total_time*100, "%")
    print("Time likelihoods", time_likelihoods, " meaning ", time_likelihoods/total_time*100, "%")
    print("Time simulating", time_simulating, " meaning ", time_simulating/total_time*100, "%")
    print("Time left", time_external, " meaning ", time_external/total_time*100, "%")
    
'''
A function plotting the RMSE curves across N for different values of y_range
given a certain method and cross trial table
'''
def plot_for_Y_range(cross_trial_table, scale=10,method="Base",
                     xlab="Number of Particles", ylab="RMSE",
                     title_beginning="RMSE for $\Delta_y$ and N",
                     N_values=[10,18,30,56,100,180,300,560,1000,1800,3000],
                     y_factor=[2,3,4,6,8,10], alpha=0, save=None):
    if not (scale==10 or scale==100):
        exit("WRONG SCALE")
    y_proxies=np.array(y_factor)*scale
    fig,ax=plt.subplots()
    if method=="Base":
        y=np.mean(cross_trial_table[:,0,:], axis=1)
    else:
        y=np.mean(cross_trial_table, axis=2)
    ax.plot(N_values,y)
    if not method=="Base":
        ax.legend(y_proxies)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_ylim(bottom=0)
    ax.set_xscale('log')
    if save==None:
        if not method=="Conditional":
            ax.set_title(title_beginning+', '+method)
            plt.savefig('figs/rmse_'+str(scale)+'_'+method+'.png')
        else:
            ax.set_title(title_beginning+', alpha = '+str(alpha))
            plt.savefig('figs/entrop_rmse_'+str(scale)+'_0'+str(alpha*100)+'.png')
    else:
        if not method=="Conditional":
            ax.set_title(title_beginning+', '+method)
            plt.savefig('figs/'+save)
        else:
            ax.set_title(title_beginning+', alpha = '+str(alpha))
            plt.savefig('figs/'+save)
        
    
'''
Computes the avg free method RMSE and correlations for a certain nb of runs
'''
def compute_free_RMSE(runs=200):
    global replacing;
    replacing=True
    free_RMSE=0
    free_Corr=np.zeros(3)
    free_Corr_delta=np.zeros((2,3))
    for i in tqdm(range(runs)):
        set_particles_start(start_type="Around")
        simulate_particles(0, K=T, make_noisy=False)
        RMSE,Corr,Corr_delta=evaluate_reconstruction()
        free_RMSE+=RMSE
        free_Corr+=Corr
        free_Corr_delta+=Corr_delta
    return free_RMSE/runs, free_Corr/runs, free_Corr_delta/runs

################################TESTS PART#####################################

#MAKE RESULT DIRECTORIES
result_dirs=['figs','preload']
for directory in result_dirs:
    if not os.path.isdir(directory):
        os.mkdir(directory)
        
#Before running any method, in order to get the plot of Lorenz and equilibria
plot_particles(0,5000,put_equilibria=True, title="Lorenz63 plot + equilibria", save_fig=True)
    
methods_tested=["Base", "Base_Interpolation", "Cumulative", "Backtrack", "Conditional", "Conditional", "Conditional"]#Methods tested
alpha_entropy_tested=[0,0,0,0,0,1,0.85]#For clarity, the first values dont matter since they are tested with methods that do not take them into account
N_values=[10,18,30,56,100,180,300,560,1000,1800,3000]#N values tested
y_factor=[2,3,4,6,8,10]#y_factors tested
scales=[10,100]#scales of x_assim tested
runs=200#number of runs on each parameter set

types_of_evaluations=['rmse','corr','corr_delta','gini','entropy']
already_done_cross=np.ones(2, dtype=bool)
all_done=True
some_done=False
for i in range(len(scales)):
    for j in range(len(types_of_evaluations)):
        table_adress='./preload/cross_trial_'+types_of_evaluations[j]+'_'+str(scales[i])+'.npy'
        if not os.path.isfile(table_adress):
            already_done_cross[i]=False
            all_done=False
    if already_done_cross[i]:
        some_done=True
            
def ask_prompt(prompt, err=False):
    if err:
        print("Please enter a valid answer, either 'y' or 'n'")
    Prompt = input(prompt)
    if not (Prompt=='y' or Prompt=='n'):
        ask_prompt(prompt, err=True)
    else:
        return Prompt

load_files=True
run_files=True
if all_done:
    prompt = "All result files are already pre-generated, they take a long time (days) to generate, do you want to load them? [y,n]"
    answer = ask_prompt(prompt)
    if answer=='n':
        load_files=False
    else:
        run_files=False
elif some_done:
    prompt = "Some result files are already pre-generated, they take a long time (days) to generate, do you want to load them? [y,n]"
    answer = ask_prompt(prompt)
    if answer=='n':
        load_files=False
else:
    load_files=False

#Initialising all tables
cross_trial_rmse_10 = np.zeros((len(methods_tested),len(N_values),len(y_factor),runs))
cross_trial_corr_10 = np.zeros((len(methods_tested),len(N_values),len(y_factor),runs,3))
cross_trial_corr_delta_10 = np.zeros((len(methods_tested),len(N_values),len(y_factor),runs,2,3))
cross_trial_gini_10 = np.zeros((len(methods_tested),len(N_values),len(y_factor),runs))
cross_trial_entropy_10 = np.zeros((len(methods_tested),len(N_values),len(y_factor),runs))
cross_trial_rmse_100 = np.zeros((len(methods_tested),len(N_values),len(y_factor),runs))
cross_trial_corr_100 = np.zeros((len(methods_tested),len(N_values),len(y_factor),runs,3))
cross_trial_corr_delta_100 = np.zeros((len(methods_tested),len(N_values),len(y_factor),runs,2,3))
cross_trial_gini_100 = np.zeros((len(methods_tested),len(N_values),len(y_factor),runs))
cross_trial_entropy_100 = np.zeros((len(methods_tested),len(N_values),len(y_factor),runs))
       
if load_files:
    print()
    print("Loading available files...")
    print()
    for i in range(len(scales)):
        for j in range(len(types_of_evaluations)):
            table_adress='./preload/cross_trial_'+types_of_evaluations[j]+'_'+str(scales[i])+'.npy'
            if os.path.isfile(table_adress):
                exec('cross_trial_'+types_of_evaluations[j]+'_'+str(scales[i])+' = np.load("'+table_adress+'")')
    print("All available files loaded.")
    print()

if run_files:
    print()
    print("Generating missing files...")
    print()
    for i in range(len(scales)):
        if not already_done_cross[i]:
            print(str("Generating files for Delta_x = "+str(scales[i])))
            
            #Setting x_assim details
            x_assim=scales[i]
            T=240*x_assim
            x_blocks = int(T/x_assim)
            std[0]=2/m.sqrt(x_assim/10)            
            dyna_std=std[0]/x_assim
            reference_simulation=lorenz_K_steps(K=T)[0]
            
            for k in range(len(methods_tested)):
                method=methods_tested[k]
                if method=="Conditional":
                    print("Testing Conditional method entropy with alpha = "+str(alpha_entropy_tested[k])+"...")
                else:
                    print("Testing "+method+" method...")
                time.sleep(0.5)
                execution_string_append=('')
                alpha_entropy=alpha_entropy_tested[k]
                for j in range(len(types_of_evaluations)):
                    execution_string_append=execution_string_append+str('cross_trial_'+types_of_evaluations[j]+'_'+str(scales[i])+'['+str(k)+']')
                    if j!=len(types_of_evaluations)-1:
                        execution_string_append=execution_string_append+','
                    else:
                        execution_string_append=execution_string_append+'='
                exec(execution_string_append+'cross_trial(method=method, N_range=N_values,y_prox=y_factor, runs=runs)')
        #Save figures
        for j in range(len(types_of_evaluations)):
            tab='cross_trial_'+types_of_evaluations[j]+'_'+str(scales[i])
            tab_adress=os.path.join("preload", tab)
            exec('np.save("'+tab_adress+'", '+tab+')')
        print()
        
    print("Missing files generated.")
    print()

#RMSE Plots
for i in range(len(scales)):
    for k in range(len(methods_tested)):
        tab='cross_trial_rmse_'+str(scales[i])+'['+str(k)+']'
        exec('plot_for_Y_range('+tab+', scale=scales[i],method=methods_tested[k], alpha=alpha_entropy_tested[k])')
        
#Entropy Plots
for i in range(len(scales)):
    tab='cross_trial_entropy_'+str(scales[i])+'[2]'
    save='entrop_cumul_'+str(scales[i])+'.png'
    exec('plot_for_Y_range('+tab+', ylab="Entropy",scale=scales[i],method=methods_tested[2],title_beginning="Entropy averages for $\Delta_y$ and N",save="'+save+'")')
    
#For Gini Plots
#Entropy Plots
for i in range(len(scales)):
    tab='cross_trial_gini_'+str(scales[i])+'[2]'
    save='gini_cumul_'+str(scales[i])+'.png'
    exec('plot_for_Y_range('+tab+', ylab="Gini Index",scale=scales[i],method=methods_tested[2],title_beginning="Gini averages for $\Delta_y$ and N",save="'+save+'")')
    
particles_shown=[0,4,8]
print("Shows correlations for particles: ", end='')
print(np.array(N_values)[particles_shown], end='')
print(" for columns and on x, y and z for lines (with Delta_y=2*Delta_x")
print()
for i in range(len(scales)):
    print("Checking for Delta_x = "+str(scales[i]))
    for k in range(len(methods_tested)):
        print("Correlation on dt time series "+ methods_tested[k], end='')
        if methods_tested[k]=="Conditional":
            print(" alpha entropy "+str(alpha_entropy_tested[k]))
        else:
            print()
        tab='cross_trial_corr_'+str(scales[i])+'['+str(k)+']'
        exec("print(np.mean("+tab+",axis=2)[particles_shown,0,:].T)")
        print()
    print()
print()
        
particles_shown=[0,4,8]
print("Shows correlations for particles: ", end='')
print(np.array(N_values)[particles_shown], end='')
print(" for columns and on x, y and z for lines (with Delta_y=2*Delta_x)")
print()
for i in range(len(scales)):
    print("Checking for Delta_x = "+str(scales[i]))
    for k in range(len(methods_tested)):
        print("Correlation on Delta_x time series "+ methods_tested[k], end='')
        if methods_tested[k]=="Conditional":
            print(" alpha entropy "+str(alpha_entropy_tested[k]))
        else:
            print()
        tab='cross_trial_corr_delta_'+str(scales[i])+'['+str(k)+']'
        exec("print(np.mean("+tab+",axis=2)[particles_shown,0,0,:].T)")
        print()
    print()
print()
        
