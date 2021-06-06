"""
创建时间特征
"""

def gen_time_features(df):
    """
    时间特征提取
    """
    df['TimeStample'] = df.index
    df['doy'] = df['TimeStample'].apply(lambda x: x.dayofyear)
    df['day'] = df['TimeStample'].apply(lambda x: x.day)
    df['month'] = df['TimeStample'].apply(lambda x: x.month)
    df['dow'] = df['TimeStample'].apply(lambda x: x.dayofweek)
    df['woy'] = df['TimeStample'].apply(lambda x: x.weekofyear)
    df['month_start'] = df['TimeStample'].apply(lambda x: x.is_month_start)
    df['month_end'] = df['TimeStample'].apply(lambda x: x.is_month_end)
    df['quarter_start'] = df['TimeStample'].apply(lambda x: x.is_quarter_start)
    df['quarter_end'] = df['TimeStample'].apply(lambda x: x.is_quarter_end)
    df['year_start'] = df['TimeStample'].apply(lambda x: x.is_year_start)
    df['year_end'] = df['TimeStample'].apply(lambda x: x.is_year_end)
    df['diff_1'] = df['Qi'].diff(1)
    df['diff_2'] = df['diff_1'].diff(1)
    del df['TimeStample']
    return df
