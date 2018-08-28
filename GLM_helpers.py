import statsmodels.api as sm
import numpy as np
import pandas as pd
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras import Sequential
from keras.layers import Dense, Lambda

# Inputs:
# stimulus: video shown to the mouse in some form
# spikes: spikes recorded from each neuron in some form
# d_stim: length of stimulus filters in samples
# d_stim: length of spike filters in samples
# f: string for the firing rate function used. options are {'exp'}
# priors: TODO
# L1: regularization parameter for sparse synapses TODO

# Returns:
# GLM network model with parameters fit

def GLM_network_fit(stimulus,spikes,d_stim, d_spk,bin_len,f='exp',priors=None,L1=None):
    N = spikes.shape[0]
    M = stimulus.shape[0]
    K = np.empty((N,M,d_stim)) # stimulus filters
    W = np.empty((N,N,d_spk))  # spike train filters
    b = np.empty((N,)) # biases
    fs = {'exp':K.exp}
    for i in range(N):
        [y, Xdsn] = construct_GLM_mat(stimulus,spikes,i, d_stim, d_spk)
        print np.sum(y), np.sum(np.sum(Xdsn))
        y = y.reshape((-1,1))
        # construct GLM model and return fit
        model = Sequential()
        model.add(Dense(1,input_dim = Xdsn.shape[1],use_bias=True))
        model.add(Lambda(lambda x: fs[f](x)*bin_len))
        model.compile(loss = 'poisson',optimizer = keras.optimizers.adam(lr=2e-1))
        model.fit(x=Xdsn,y=y,epochs=150,batch_size = 1000000)
        p = model.get_weights()[0]
        F[i,:,:] = p[:M*d_stim].reshape((M,d_stim))
        W[i,:,:] = p[M*d_stim:].reshape((N,d_spk))
        b[i] = model.get_weights()[1]
    return (F,W,b)


# Inputs:
# data_set: EphysObservatory data_set structure
# bin_len: duration of a bin in seconds
# t_start: time to start counting spikes
# t_final: time to stop counting spikes
# probes: list of strings of probe names to be used (default is all probes)
# regions: list of strings of brain regions to be used (default is all regions)
def bin_spikes(data_set,bin_len,t_start,t_final,probes=None,regions=None):
    if probes is None:
        probes = data_set.probe_list
    if regions is None:
        regions = data_set.unit_df.structure.unique()
    
    #gather cells from desired regions and probes into cell_table
    use_cells = False
    for probe in probes:
        for region in regions:
            use_cells |= (data_set.unit_df.probe==probe) & (data_set.unit_df.structure==region)
    cell_table = data_set.unit_df[use_cells]
    
    N = len(cell_table)     #number of cells
    T = int(np.floor((t_final-t_start)/bin_len)) #number of time bins
    binned_spikes = np.zeros((N,T)) # binned_spikes[i,j] is the number of spikes from neuron i in time bin j

    #for each cell in the table, add each spike to the appropriate bin
    i = 0
    for z,cell in cell_table.iterrows(): 
        for spike_time in data_set.spike_times[cell['probe']][cell['unit_id']]:
            t = int(np.floor((spike_time-t_start)/bin_len))
            if (t >=0) & (t<T):
                binned_spikes[i,t] += 1
        i+=1    
    return (binned_spikes, cell_table)


# Inputs
# flat_stimulus: M x T matrix of stimuli
# binned_spikes: N x T matrix of spike counts
# i: index of the neuron we're constructing the matrix for
# d_stim: duration of stimulus filter (# time bins)
# d_spk: duration of spike filters (# time bins)
def construct_GLM_mat(flat_stimulus, binned_spikes, i, d_stim, d_spk):
    (N,T) = binned_spikes.shape # N is number of neurons, T is number of time bins
    (M,T) = flat_stimulus.shape # M is the size of a stimulus
    X_dsn = np.empty((T-d_stim,M*d_stim+N*d_spk))
    d_max = max(d_stim,d_spk)
    y = np.empty((T-d_max,))
    for t in range(T-d_max):
        y[t] = binned_spikes[i,t+d_max]
        X_dsn[t,:M*d_stim] = np.fliplr(flat_stimulus[:,t+d_max-d_stim:t+d_max]).reshape((1,-1))  #stimulus inputs
        X_dsn[t,M*d_stim:] = np.fliplr(binned_spikes[:,t+d_max-d_spk:t+d_max]).reshape((1,-1)) #spike inputs
    return (y, X_dsn)      



    
