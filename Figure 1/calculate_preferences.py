def calc_pref(df, condition=None):
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