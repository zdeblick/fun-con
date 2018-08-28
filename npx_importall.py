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


# aggregate all unit dataframe in the same order
unit_df_all=dict()
for i in range(len(expt_info_df)): # index to row in multi_probe_expt_info
    # List of structures from which units were recorded
    for region_name in np.unique(data_set_all[i].unit_df['structure']):
        if region_name in unit_df_all.keys():
            datasubset=data_set_all[i].unit_df[data_set_all[i].unit_df['structure']==region_name].copy()
            pds=pd.Series([i] * datasubset.shape[0])
            datasubset=datasubset.assign(data_set_ind = pds.values)
            unit_df_all[region_name]=unit_df_all[region_name].append(datasubset)
#             print(i,region_name,unit_df_all[region_name].shape)
        else:
            datasubset=data_set_all[i].unit_df[data_set_all[i].unit_df['structure']==region_name].copy()
            pds=pd.Series([i] * datasubset.shape[0])
            datasubset=datasubset.assign(data_set_ind = pds.values)
            unit_df_all[region_name]=datasubset
#             print(i,region_name,unit_df_all[region_name].shape)


# aggregate all unit spike times in the same order
spike_times_all=dict()
for region in unit_df_all.keys():
    if 'index' not in unit_df_all[region].keys():
        unit_df_all[region]=unit_df_all[region].reset_index() #(drop=True)
    spike_times_all[region]=[]
    for i in range(unit_df_all[region].shape[0]):
        whichdataset = unit_df_all[region]['data_set_ind'][i]
        whichprobe = unit_df_all[region]['probe'][i]
        whichunit = unit_df_all[region]['unit_id'][i]
        spike_times_all[region].append(data_set_all[whichdataset].spike_times[whichprobe][whichunit])
#     probe_spikes = data_set.spike_times[probes[k]]



# aggregate all spike waveforms in the same order
spike_waveforms_all=[]
for i in range(len(expt_info_df)): 
    spike_waveforms_all.append( data_set_all[i].get_waveforms() )

spikewf_all=dict()
spikewftrough_all=dict()
spikewfpeak_all=dict()
for region in unit_df_all.keys():
    if region == 'V1': # single probe experiments do not have waveform information as of yet
        continue
    spikewf_all[region]=np.zeros([unit_df_all[region].shape[0], 82])   
    spikewftrough_all[region]=np.zeros([unit_df_all[region].shape[0], 1])   
    spikewfpeak_all[region]=np.zeros([unit_df_all[region].shape[0], 1])   
    for i in range(unit_df_all[region].shape[0]):
        whichdataset = unit_df_all[region]['data_set_ind'][i]
        whichprobe = unit_df_all[region]['probe'][i]
        whichunit = unit_df_all[region]['unit_id'][i]

        spikewf_all[region][i,:]=spike_waveforms_all[whichdataset][whichprobe][whichunit]
        spikewftrough_all[region][i]=spike_waveforms_all[whichdataset][whichprobe][whichunit].argmin()
        spikewfpeak_all[region][i]=spike_waveforms_all[whichdataset][whichprobe][whichunit].argmax()


for region in unit_df_all.keys():
    if region == 'V1': # single probe experiments do not have waveform information as of yet
        continue    
    if 'RSFS' not in unit_df_all[region].keys():
        spikewfp2t=(spikewfpeak_all[region]-spikewftrough_all[region])/30.
        #rsfs=np.ndarray(shape=[unit_df_all[region].shape[0],1], dtype='str')
        rsfs=np.array(['SU']*unit_df_all[region].shape[0])
        rsinds,_=np.where(spikewfp2t>0.4)
        rsfs[rsinds]='RS'
        fsinds,_=np.where(spikewfp2t<=0.4)
        rsfs[fsinds]='FS'
        srsfs=pd.Series(rsfs)
        print(region, unit_df_all[region].shape, srsfs.shape)
        unit_df_all[region]=unit_df_all[region].assign(RSFS = srsfs.values)
    print(unit_df_all[region].keys())


# nonvisregions=[x for x in mp_data_set[1].unit_df['structure'].unique() if not x in mp_data_set[1].region_list]
# print(nonvisregions)



