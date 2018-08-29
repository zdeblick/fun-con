drive_path = '/data/dynamic-brain-workshop/visual_coding_neuropixels'
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
%matplotlib inline

# Provide path to manifest file
manifest_file = os.path.join(drive_path,'ephys_manifest.csv')

# Create a dataframe 
expt_info_df = pd.read_csv(manifest_file)

# Make new dataframe by selecting only single-probe experiments
single_probe_expt_info = expt_info_df[expt_info_df.experiment_type == 'single_probe']
print('Number of single-probe experiments: %s') %len(single_probe_expt_info)

#make new dataframe by selecting only multi-probe experiments
multi_probe_expt_info = expt_info_df[expt_info_df.experiment_type == 'multi_probe']
print('Number of multi-probe experiments: %s') %len(multi_probe_expt_info)

# Import NWB_adapter
from swdb_2018_neuropixels.ephys_nwb_adapter import NWB_adapter

# load all dataset objects
data_set_all=[]
for i in range(len(expt_info_df)): # index to row in multi_probe_expt_info
    expt_filename  = expt_info_df.iloc[i]['nwb_filename']
    # iloc allows you to access particular rows of dataset
    print expt_filename

    # Specify full path to the .nwb file
    nwb_file = os.path.join(drive_path, expt_filename)

    data_set_all.append(NWB_adapter(nwb_file)) # this linetakes a while
    
    print(data_set_all[i].number_cells)  #"tab completion"


# aggregate all unit dataframe 
for i in range(len(expt_info_df)): # index to row in multi_probe_expt_info
    unit_df_subset=data_set_all[i].unit_df.copy()
    pds0=pd.Series([i] * unit_df_subset.shape[0])
    unit_df_subset=unit_df_subset.assign(data_set_ind = pds0.values)
    pds1=pd.Series([expt_info_df.iloc[i]['nwb_filename']] * unit_df_subset.shape[0])
    unit_df_subset=unit_df_subset.assign(nwb_filename = pds1.values)
    if i==0:
        unit_df_all=unit_df_subset
    else:
        unit_df_all=unit_df_all.append(unit_df_subset)

if 'index' not in unit_df_all.keys():
    unit_df_all=unit_df_all.reset_index() #(drop=True)


# aggregate all spike waveforms 
spike_waveforms_all=[] # empty list
for i in range(len(expt_info_df)): 
    spike_waveforms_all.append( data_set_all[i].get_waveforms() )

spikewf_all=np.zeros([unit_df_all.shape[0], 82])
spikewftrough_all=np.zeros([unit_df_all.shape[0], 1])   
spikewfpeak_all=np.zeros([unit_df_all.shape[0], 1])   
for i in range(unit_df_all.shape[0]):
    whichdataset = unit_df_all['data_set_ind'][i]
    whichprobe = unit_df_all['probe'][i]
    whichunit = unit_df_all['unit_id'][i]
    
    if whichprobe == 'V1':
        spikewf_all[i,:]=np.nan
        spikewftrough_all[i]=np.nan
        spikewfpeak_all[i]=np.nan       
    else:
        spikewf_all[i,:]=spike_waveforms_all[whichdataset][whichprobe][whichunit]
        spikewftrough_all[i]=spike_waveforms_all[whichdataset][whichprobe][whichunit].argmin()
        spikewfpeak_all[i]=spike_waveforms_all[whichdataset][whichprobe][whichunit].argmax()


if 'RSFS' not in unit_df_all.keys():
    spikewfp2t=(spikewfpeak_all-spikewftrough_all)/30.
    rsfs=np.array(['SU']*unit_df_all.shape[0])
    rsinds,_=np.where(spikewfp2t>0.4)
    rsfs[rsinds]='RS'
    fsinds,_=np.where(spikewfp2t<=0.4)
    rsfs[fsinds]='FS'
    srsfs=pd.Series(rsfs)
    unit_df_all=unit_df_all.assign(RSFS = srsfs.values)
    print(unit_df_all.keys())


# aggregate all spike time in the same order as unit_df_all
spike_times_all=[] # empty list
for i in range(unit_df_all.shape[0]):
    whichdataset = unit_df_all['data_set_ind'][i]
    whichprobe = unit_df_all['probe'][i]
    whichunit = unit_df_all['unit_id'][i]
    spike_times_all.append(data_set_all[whichdataset].spike_times[whichprobe][whichunit])
