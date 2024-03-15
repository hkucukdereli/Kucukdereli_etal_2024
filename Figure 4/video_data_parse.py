
import numpy as np
import pandas as pd
import sys
import socket
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import seaborn as sns
import sklearn
from tqdm.auto import tqdm
#from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from PIL import Image 
import os
serverpath  = fr'\\anastasia\data' if socket.gethostname() == 'SANTIAGO' else '/mnt/anastasia/data'

sys.path.append(f"{serverpath}/behavior/hakan/virtually/")

import virtually
#vr.basedir = f"{serverpath}/behavior/hakan"


def load_data(df_video, query, debug=False):
    ''''
    if debug == True I'll test if I can detect if the model can just predict running vs sitting
    '''
    vr = df_video.query(query).data.iloc[0]
    
    if 'test' not in  query:
        if debug == True:
            raise NotImplementedError
        else:
            stim_frames = vr.table.query('stim_1>0  & dwell>0').index # Sitting frames  + stim!
            nat_frames = vr.table.query('neutral_1>0 & dwell>0').index # Sitting frames + natural
    else:
        if debug == True:
            stim_frames = vr.table.query('(trial_1>0 | trial_2>0) & explore>0').index # running
            nat_frames = vr.table.query('(trial_1>0 | trial_2>0) & dwell>0').index # dwell
        else:
            stim_frames = vr.table.query('(stim_1>0 | stim_2>0) & dwell>0').index # Sitting frames  + stim!
            nat_frames = vr.table.query('(neutral_1>0 | neutral_2>0) & dwell>0').index # Sitting frames + natural

    nat_frames = set(nat_frames) - set(stim_frames)
    
    # remove frames that do not exists in relevant_data
#     non_existing_frames = set(range(max(relevant_data.index),max(relevant_data.index)*5))
#     nat_frames = set(nat_frames) - non_existing_frames
#     stim_frames = set(stim_frames) - non_existing_frames
    
    relevant_data = pd.DataFrame({'stim':np.zeros(len(vr.table))})
    data = relevant_data.loc[np.sort(list(stim_frames) + list(nat_frames))].copy()
    data['stim'] = -1
    data.loc[list(stim_frames),'stim'] = 1
    data.loc[list(nat_frames),'stim'] = 0

    assert np.all(set(data['stim'].unique()) == set([1,0]))
    return data



def parse_data_frame_ids(data,rnd_state=30,min_diff=100, chunk_min_size=3):
    '''
    samp state how many sample to merge, e.g.:
    [1,2,3,4,5,6] -- [1,2,3],[2,3,4],[3,4,5],[4,5,6]
    
    '''
    rnd_state=rnd_state
    #rus = RandomUnderSampler()#random_state=rnd_state)

    train_p = 0.8
    test_p = 0.2
    rnd = np.random.RandomState(rnd_state)

    frames_chunks_ids = []
    frames_chunks_ids_labels = []
    last_i = 0 
    for i in np.nonzero(np.diff(data.index)>min_diff)[0]:
        frames_chunks_ids.append(list(data.index[last_i:i+1]))
        frames_chunks_ids_labels.append(list(data.iloc[last_i:i+1].stim))
        last_i=i+1


    train_inds = []
    test_inds = []
    total_frames_size = len([j for i in frames_chunks_ids for j in i])

    len_1_train = 0
    len_0_train = 0
    len_1_test = 0
    len_0_test = 0

    do_what = 'train'
    for i in rnd.choice(range(len(frames_chunks_ids)),size=len(frames_chunks_ids),replace=False):
        sample_ident = int(np.round(np.mean(frames_chunks_ids_labels[i]))) # maybe not all samples are 1
        if len(frames_chunks_ids[i])<chunk_min_size:
            continue
        if sample_ident == 0 :
            if len_0_train*(test_p/train_p)<len_0_test:
                train_inds.append(frames_chunks_ids[i])
                len_0_train+=len(frames_chunks_ids[i])
            else:
                test_inds.append(frames_chunks_ids[i])
                len_0_test+=len(frames_chunks_ids[i])
        else:
            if len_1_train*(test_p/train_p)<len_1_test:
                train_inds.append(frames_chunks_ids[i])
                len_1_train+=len(frames_chunks_ids[i])
            else:
                test_inds.append(frames_chunks_ids[i])
                len_1_test+=len(frames_chunks_ids[i])
    
    data['train'] = None # This has to be None!!
    for j,chunk in enumerate(train_inds):
        data.loc[chunk,'train'] = 1
        data.loc[chunk,'chunk'] = j
        
    for j,chunk in enumerate(test_inds):
        data.loc[chunk,'train'] = 0
        data.loc[chunk,'chunk'] = j
        
    return data.dropna()



def parse_data_frame_ids_start_end(data):
    '''
    samp state how many sample to merge, e.g.:
    [1,2,3,4,5,6] -- [1,2,3],[2,3,4],[3,4,5],[4,5,6]
    
    '''
    rnd_state=30
    #rus = RandomUnderSampler()#random_state=rnd_state)

    train_p = 0.8
    test_p = 0.2
    data['train'] = 1
    data['chunk'] = -1
    data.iloc[:int(len(data)*train_p) ,data.columns.get_loc('train')] = 1
    data.iloc[:int(len(data)*train_p) ,data.columns.get_loc('chunk')] = 0
    
    data.iloc[int(len(data)*train_p): ,data.columns.get_loc('train')] = 0
    data.iloc[int(len(data)*train_p): ,data.columns.get_loc('chunk')] = 1

    return data
    

base_path_cache = '/mnt/ssd_cache/manual_cache/'
def get_data_samples(expes,debug=False,min_diff=100,chunk_min_size=3, data_split ='chunks', force_days=None):
    '''
    expes = ['test_','test',preference], ['test_']
    force_days = can be a list of days to force to be loaded, e.g. [1,2,3] 
    '''

    #if socket.gethostname() == 'colab02':
    #    df_video = pd.read_pickle(f"/home/oamsalem/test_data_on_ssd/no-stress_first_table_female.npy")
    #else:
    try:
        df_video = pd.read_pickle(f'{base_path_cache}/oren/no-stress_first_table_female.npy')
    except:
        df_video = pd.read_pickle(f"{serverpath}/behavior/hakan/oren/no-stress_first_table_female.npy")
        if 'colab' in socket.gethostname():
            os.makedirs(f'{base_path_cache}/oren', exist_ok=True)
            df_video.to_pickle(f'{base_path_cache}/oren/no-stress_first_table_female.npy')    

    datas = []
    for expe in expes:
        if force_days is None:
            days = [1,2] if expe == 'preference' else [1,2,3,4,5]
        else:
            days = force_days

        to_pnd = {'mouse':[],'day':[],'test_stim_accuracy':[],'train_stim_accuracy':[], 'sample_test':[],'sample_train':[]}
        for mouse in tqdm(list(set(df_video.query("sex=='female'").mouse))):
            for day in days:
                data = load_data(df_video, query = f"mouse=='{mouse}' and experiment=='{expe}' and day=={day}", debug=debug).copy()
                if data is None:
                    continue
                if data_split == 'chunks':
                    data = parse_data_frame_ids(data,min_diff=min_diff,chunk_min_size=chunk_min_size).dropna()
                elif data_split == 'start_end':
                    data = parse_data_frame_ids_start_end(data)
                data['mouse'] = mouse
                data['day'] = day
                data['date'] = df_video.query(f'mouse=="{mouse}" and day=={day} and experiment=="{expe}"').date.values[0]
                datas.append(data)
            
    full_df = pd.concat(datas)
    return full_df