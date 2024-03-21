import numpy as np
import pandas as pd

def calcPref(df, condition=None):
    """
    Calculate preference value for a given locomotion state. Conditions can be `dwell`, `explore`, `transition` or None for any locomotion state.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe containing the data.
    condition : str, optional
        Locomotion state to use. The default is None. 
        Condition can be `dwell`, `explore`, `transition` or None for any locomotion state.

    Returns
    -------
    pref_val : float
        Preference value.

    """
    if condition:
        stim = df.query(f'session in [1,2] & stim!=0 & {condition}==True').dt.sum() 
        neutral = df.query(f'session in [1,2] & neutral!=0 & {condition}==True').dt.sum() 
    else:
        stim = df.query(f'session in [1,2] & stim!=0').dt.sum() 
        neutral = df.query(f'session in [1,2] & neutral!=0').dt.sum() 

    pref_val = (stim - neutral) / (stim + neutral)
    
    return pref_val

def calcPrefShuffle(df, nshuffles, condition=None):
    """
    Calculate preference values from a shuffled distribution. Conditions can be `dwell`, `explore`, `transition` or None for any locomotion state.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe containing the data.
    nshuffles : int
        Number of shuffles.
    condition : str, optional
        Locomotion state to use. The default is None. 
        Condition can be `dwell`, `explore`, `transition` or None for any locomotion state.

    Returns
    -------
    pref_list : list
        List of preference values.

    """
    if condition:
        stim = df.query(f'session in [1,2] & stim!=0 & {condition}==True').groupby('cueid').dt.sum()
        neutral = df.query(f'session in [1,2] & neutral!=0 & {condition}==True').groupby('cueid').dt.sum()
    else:
        stim = df.query(f'session in [1,2] & stim!=0').groupby('cueid').dt.sum()
        neutral = df.query(f'session in [1,2] & neutral!=0').groupby('cueid').dt.sum()

    pref_list = []
    for n in range(nshuffles):
        stim_t = stim.sample(n=len(stim), replace=True, random_state=n).sum()
        neutral_t = neutral.sample(n=len(neutral), replace=True, random_state=n).sum()

        pref_val = (stim_t - neutral_t) / (stim_t + neutral_t)
        pref_list.append(pref_val)
    
    return np.sort(np.array(pref_list))

def calcPrefSession(df, session, condition=None):
    """
    Calculate preference value for a given locomotion state for a given session. Conditions can be `dwell`, `explore`, `transition` or None for any locomotion state.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe containing the data.
    session : int
        Session number. Sessions number can be 1 or 2 for the first and last 30 min of the RTPP experiment.
    condition : str, optional
        Locomotion state to use. The default is None. 
        Condition can be `dwell`, `explore`, `transition` or None for any locomotion state.

    Returns
    -------
    pref_val : float
        Preference value.

    """
    if condition:
        stim = df.query(f'session==@session & stim!=0 & {condition}==True').dt.sum() 
        neutral = df.query(f'session ==@session & neutral!=0 & {condition}==True').dt.sum() 
    else:
        stim = df.query(f'session==@session & stim!=0').dt.sum() 
        neutral = df.query(f'session==@session & neutral!=0').dt.sum() 

    pref_val = (stim - neutral) / (stim + neutral)
    
    return pref_val

def calcPrefSessionShuffle(df, session, nshuffles, condition=None):
    """
    Calculate preference values from a shuffled distribution for a given Session. Conditions can be `dwell`, `explore`, `transition` or None for any locomotion state.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe containing the data.
    session : int
        Session number. Sessions number can be 1 or 2 for the first and last 30 min of the RTPP experiment.
    nshuffles : int
        Number of shuffles.
    condition : str, optional
        Locomotion state to use. The default is None. 
        Condition can be `dwell`, `explore`, `transition` or None for any locomotion state.

    Returns
    -------
    pref_list : list
        List of preference values.

    """
    if condition:
        stim = df.query(f'session==@session & stim!=0 & {condition}==True').groupby('cueid').dt.sum()
        neutral = df.query(f'session==@session & neutral!=0 & {condition}==True').groupby('cueid').dt.sum()
    else:
        stim = df.query(f'session==@session & stim!=0').groupby('cueid').dt.sum()
        neutral = df.query(f'session==@session & neutral!=0').groupby('cueid').dt.sum()

    pref_list = []
    for n in range(nshuffles):
        stim_t = stim.sample(n=len(stim), replace=True, random_state=n).sum()
        neutral_t = neutral.sample(n=len(neutral), replace=True, random_state=n).sum()

        pref_val = (stim_t - neutral_t) / (stim_t + neutral_t)
        pref_list.append(pref_val)
    
    return np.sort(np.array(pref_list))