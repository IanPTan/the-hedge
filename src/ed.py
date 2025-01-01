from ticker import *


def spike(article_df, df_original, period=30, mincheck=1, IQR_multiplier=1.5, lookback_bounds=15, lookback_normalize=15, normalize_multiplier=3):

    mincheck += 1
    df = df_original.copy()
    df.index = df_original['time']

    article_df["spike"] = np.nan
    article_df["spike-type"] = np.nan
    article_df["peak"] = np.nan
    article_df['normalize'] = np.nan

    for i in range(len(article_df)):

        ## Find hour holding the stock data
        unix_time = article_df.loc[i, 'time']
        if unix_time > df.index[-1] or unix_time < df.index[0]:
            continue
        
        
        dtime = pd.to_datetime(unix_time, unit='s')
        dtime = dtime - timedelta(minutes=dtime.minute % period, seconds=dtime.second, microseconds=dtime.microsecond)       
        
        ## If article is released during trading hours or not
        intrading = int(dtime.timestamp()) in df.index

        times = df[df['time'] <= int(dtime.timestamp())]
        dtime = times.index[-1]


        ## Find subset
        ## Since 30 minute intervals
        prev = df[df['time'] < dtime]
        prev = prev.tail(lookback_bounds)

        close = prev['close']
        close = close.sort_values()

        q3 = close.quantile(0.75)

        q1 = close.quantile(0.25)
        
        IQR = q3 - q1

        upper_bound =  q3 + IQR_multiplier * IQR
        lower_bound =  q1 - IQR_multiplier * IQR

        after_article = df[df['time'] >= dtime]
        in_mincheck = after_article.head(mincheck)

        if not intrading:
            in_mincheck = in_mincheck[1:]

        highest = in_mincheck.loc[in_mincheck['high'].idxmax()]
        lowest = in_mincheck.loc[in_mincheck['low'].idxmin()]

        if highest.loc['high'] > upper_bound and lowest.loc['low'] < lower_bound:
            article_df.loc[i, 'spike'] = True
            article_df.loc[i, 'spike-type'] = 0
        elif highest.loc['high'] > upper_bound:
            article_df.loc[i, 'spike'] = True
            article_df.loc[i, 'spike-type'] = 1
            article_df.loc[i, 'peak'] = highest['time']
        elif lowest.loc['low'] < lower_bound:
            article_df.loc[i, 'spike'] = True
            article_df.loc[i, 'spike-type'] = -1
            article_df.loc[i, 'peak'] = lowest['time']
        else:
            article_df.loc[i, 'spike'] = False
            continue
    

        past = df[df.index < dtime]
        past  = past.tail(lookback_normalize)
        moving_set = pd.concat([past['open'], past['close']], ignore_index=True)
        moving_avg = moving_set.mean()
        moving_sd = moving_set.std()       
        
        normal = after_article[abs(after_article['close'] - moving_avg) < normalize_multiplier *  moving_sd]
        first_normalize = normal.head(1)

        if first_normalize.empty:
            article_df.loc[i, 'normalize'] = False 
        else:
            article_df.loc[i, 'normalize'] = first_normalize.index[0]
    
    return article_df
