def calcPref(df, condition=None):
    """
    Calculate preference value for a given locomotion state. Conditions can be `dwell`, `explore`, `transition` or None for any locomotion state.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe containing the data.
    condition : str, optional
        Condition to filter the data. The default is None.

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
        Condition to filter the data. The default is None.

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