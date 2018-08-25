import statsmodels.api as sm

# Inputs:
# stimulus: video shown to the mouse in some form
# spikes: spikes recorded from each neuron in some form
# link: string for the link function used. options are {'log', 'logit'}
# priors: TODO
# L1: regularization parameter for sparse synapses TODO

# Returns:
# GLM network model with parameters fit

def GLM_network_fit(stimulus,spikes,link='log',priors=None,L1=None):
    # flatten stimulus and spikes into design matrix and observation matrix
    Xdsn, y = construct_GLM_mats(stimulus,spikes)
    
    # construct GLM model and return fit
    links = {'log':sm.genmod.families.links.log, 'logit':sm.genmod.families.links.logit}
    
    if priors is None and L1 is None:
        glm_pois = sm.GLM(sm.add_constant(y), sm.add_constant(Xdsn), family=sm.families.Poisson(link=links[link]))
        return glm_pois.fit()
    else:
        return None
    
    
def construct_GLM_mats(flat_stimulus, binned_spikes):
    
# Inputs:
# data_set: EphysObservatory data_set structure
# bin_len: duration of a bin in seconds
# t_start: time to start counting spikes
# t_final: time to stop counting spikes
# probes: list of strings of probe names to be used (default is all probes)
# regions: list of strings of brain regions to be used (default is all regions)
def bin_spikes(data_set,bin_len,t_start,t_final,probes=data_set.probe_list,regions=data_set.unit_df.structure.unique()):
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
            t = int(np.floor(spike_time/bin_len))
            if (t >=0) & (t<T):
                binned_spikes[i,t] += 1
        i+=1    
    return (binned_spikes, cell_table)

    



    
