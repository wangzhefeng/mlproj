import pandas as pd


"""
创建时间特征
"""


def gen_time_features(df, datetime_format, datetime_is_index = False, datetime_name = None, is_test = False):
    """
    时间特征提取
    """
    def applyer(row):
        """
        判断是否是周末
        """
        if row == 5 or row == 6:
            return 1
        else:
            return 0
    df[datetime_name] = pd.to_datetime(df[datetime_name], format = datetime_format)
    if datetime_is_index:
        df['DT'] = df.index
    else:
        df["DT"] = df[datetime_name]
    df["year"] = df["DT"].apply(lambda x: x.year)
    df['month'] = df['DT'].apply(lambda x: x.month)
    df['day'] = df['DT'].apply(lambda x: x.day)
    df["hour"] = df["DT"].apply(lambda x: x.hour)
    # df['doy'] = df['DT'].apply(lambda x: x.dayofyear)
    # df['woy'] = df['DT'].apply(lambda x: x.weekofyear)
    if is_test:
        pass
    else:
        df['dow'] = df['DT'].apply(lambda x: x.dayofweek)
        df['weekend'] = df['dow'].apply(applyer)
    # df['month_start'] = df['DT'].apply(lambda x: x.is_month_start)
    # df['month_end'] = df['DT'].apply(lambda x: x.is_month_end)
    # df['quarter_start'] = df['DT'].apply(lambda x: x.is_quarter_start)
    # df['quarter_end'] = df['DT'].apply(lambda x: x.is_quarter_end)
    # df['year_start'] = df['DT'].apply(lambda x: x.is_year_start)
    # df['year_end'] = df['DT'].apply(lambda x: x.is_year_end)
    # df['diff_1'] = df['Qi'].diff(1)
    # df['diff_2'] = df['diff_1'].diff(1)
    del df['DT']
    return df
