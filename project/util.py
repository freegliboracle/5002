import os
import datetime
import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder

data_path = './data/'
cache_path = './cache/'
feature_path = './feature/'

station_info = pd.read_csv(data_path+'station_info.csv')
station_info['sitetype'] = LabelEncoder().fit_transform(station_info['sitetype'])
aq_station_id = station_info['station_id'].tolist()

aq = pd.read_hdf(data_path+'aq_nomissinghour.hdf')
aq2 = aq.copy()
aq = aq.merge(station_info,on='station_id',how='left')
aq['hour'] = pd.to_datetime(aq['time']).dt.hour
aq['date'] = aq['time'].str[:10]

# date shifted by days, either forward or backward
def date_add_days(start_date, days):
    end_date = parse(start_date[:10]) + timedelta(days=days)
    end_date = end_date.strftime('%Y-%m-%d')
    return end_date

# date shifted by hours, either forward or backward
def date_add_hours(start_date, hours):
    end_date = parse(start_date) + timedelta(hours=hours)
    end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
    return end_date

# compress data
def convert_dtypes(data,predictors,silent=False):
    for c in predictors:
        if data[c].dtypes == 'O':
            try:
                data[c] = data[c].astype('float32')
            except:
                if not silent:
                    print('feature {} format cannot be converted'.format(c))
        if data[c].dtypes == 'float64':
            data[c] = data[c].astype('float32')
    return data

# convert from list to dataframe
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result

def groupby(data,stat,key,value,func):
    key = key if type(key)==list else [key]
    datatemp = data[key].copy()
    feature = stat.groupby(key,as_index=False)[value].agg({'feat':func})
    datatemp = datatemp.merge(feature,on=key,how='left')
    return datatemp['feat'].values

# return a dataframe that includes basic spatial and temporal information for each aq station
# feature will be merged to each row
# 1680 rows: 35 stations * 48 hours
def pre_treatment(data_key):
    result_path = cache_path + 'data_{}.hdf'.format(data_key)
    if os.path.exists(result_path):
        data = pd.read_hdf(result_path, 'w')
    else:
#         48 hours
        times = pd.date_range(data_key,date_add_days(data_key,2),freq='H')[:-1]
        data = pd.DataFrame(index=times,columns=aq_station_id).unstack().reset_index().drop(0,axis=1)
        data.columns = ['station_id','time']
        data = data.merge(station_info, on='station_id', how='left')
        data['hour'] = data['time'].dt.hour
        data['month'] = data['time'].dt.month
        data['year'] = data['time'].dt.year
        data['day_of_week'] = data['time'].dt.dayofweek
        data['day_of_month'] = data['time'].dt.day
        data['day_of_year'] = data['time'].dt.dayofyear
        data['time'] = data['time'].astype(str)
        data['date'] = data['time'].str[:10]
        data['diff_of_hour'] = (data['date']!=data_key).astype(int)*24+data['hour']
        data.reset_index(drop=True, inplace=True)
        data.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return data

# get feature for the previous 24 hours
# 144 features
def get_24hour_feat(data,data_key,replace):
    result_path = cache_path + '24hour_feat_{}_{}hours_ago.hdf'.format(data_key,1)
    if os.path.exists(result_path) & (not replace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        start_time = date_add_hours(data_key, -25)
        end_time = date_add_hours(data_key, -1)
#         data temp: 840(35*24) rows * 13 columns
        data_temp = aq[(aq['time'] < end_time) & (aq['time'] >= start_time)].copy()
        feat = data[['station_id', 'sitetype']].copy()
#         get value of the label for the last 24 hours
#         72 features (24*3)
        for label in ['PM2.5','PM10','O3']:
            result_temp = data_temp.set_index(['station_id', 'time'])[label].unstack()   # 35 stations * 24 hours
#             label and hour of the day
            result_temp.columns = ['{}_{}hour_last'.format(label,c[11:13]) for c in result_temp.columns]
            feat = feat.merge(result_temp.reset_index(),on='station_id',how='left')
#         get the average of the label by sitetype
#         72 features (24*3)
        for label in ['PM2.5','PM10','O3']:
            result_temp = data_temp.groupby(['sitetype', 'time'])[label].mean().unstack()
            result_temp.columns = ['{}_{}hour_last_sitetype'.format(label,c[11:13]) for c in result_temp.columns]
            feat = feat.merge(result_temp.reset_index(),on='sitetype',how='left')
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# get feature from the past n days
# n: number of days to shift
# 9 features
# need to fix the problem of 'date'
def get_nday_mean_feat(data,data_key,n,replace):
    result_path = cache_path + '{}day_mean_feat{}_{}hours_ago.hdf'.format(n,data_key,1)
    if os.path.exists(result_path) & (not replace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        start_time = date_add_hours(data_key, -1-24*n)
        end_time = date_add_hours(data_key, -1)
        data_temp = aq[(aq['time']<end_time) & (aq['time']>=start_time)]
        feat = data[['station_id','hour','date']].copy()
#         average by n days at a particular hour for each station
#         3 features
        for label in ['PM2.5','PM10','O3']:
            feat['{}day_hour_{}_mean'.format(n,label)] = groupby(feat, data_temp, ['station_id','hour'], label, np.mean)
#         compute the average of n days for each station
#         3 features
# return nan for 'date' because of mismatch of date during merging
# needs to be fixed
        for label in ['PM2.5','PM10','O3']:
            feat['{}day_{}_mean_city'.format(n,label)] = groupby(feat, data_temp, ['station_id'], label, np.mean)
#         compute the average of n days of all stations at a particular hour
#         3 features
        for label in ['PM2.5','PM10','O3']:
            feat['{}day_hour_{}_mean_city'.format(n,label)] = groupby(feat, data_temp, ['hour'], label, np.mean)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# get weather feature
# len(i)*5+6 features
def get_weather_feat(data,data_key,replace):
    result_path = cache_path + 'weather_feat{}.hdf'.format(data_key)
    if os.path.exists(result_path) & (not replace):
        feat = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data[['station_id','time']].copy()
        end_time1 = date_add_days(data_key, -1)
        end_time2 = date_add_days(data_key, 1)
        try:
            weather1 = pd.read_csv(cache_path + 'aq_meo_grid_{}.csv'.format(end_time1[:10]))
            weather2 = pd.read_csv(cache_path + 'aq_meo_grid_{}.csv'.format(end_time2[:10]))
            weather = weather1.append(weather2)
            weather.drop(['latitude', 'longitude'], axis=1, inplace=True)
            date_time = data_temp['time'].copy()
            feat_columns = weather.columns.copy()
            for i in [-18,-12,-6,-4,-3,-2,-1,0,1,2,3]:
                weather.columns = [c+'_ahead{}'.format(i) if c not in ['station_id','time'] else c for c in feat_columns]
                data_temp['time'] = date_time.apply(lambda x: date_add_hours(x,i))
                data_temp = data_temp.merge(weather,on=['station_id','time'],how='left')
            data_temp['temperature_diff_1'] = data_temp['temperature_ahead0'] - data_temp['temperature_ahead-1']
            data_temp['temperature_diff_2'] = data_temp['temperature_ahead0'] - data_temp['temperature_ahead-2']
            data_temp['temperature_diff_21'] = data_temp['temperature_ahead-1'] - data_temp['temperature_ahead-2']
            data_temp['temperature_diff_3'] = data_temp['temperature_ahead0'] - data_temp['temperature_ahead-3']
            data_temp['temperature_diff_31'] = data_temp['temperature_ahead-2'] - data_temp['temperature_ahead-3']
            data_temp['humidity_diff_1'] = (data_temp['humidity_ahead0'] - data_temp['humidity_ahead-1'])/data_temp['humidity_ahead0']
            feat = data_temp.drop(['station_id','time'],axis=1)
            feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
        except:
            feat = pd.DataFrame()
            print('{} weather data is empty'.format(end_time[:10]))
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# add labels
def get_label(result):
    return result.merge(aq2[['station_id','time','PM2.5','PM10','O3']],on=['station_id','time'],how='left')

# 12 features
def second_feat(result):
    try:
        result['PM2.5_21hour_last/PM2.5_20hour_last_rate'] = result['PM2.5_21hour_last']/result['PM2.5_20hour_last']
        result['PM10_21hour_last/PM10_20hour_last_rate'] = result['PM10_21hour_last'] / result['PM10_20hour_last']
        result['O3_21hour_last/O3_20hour_last_rate'] = result['O3_21hour_last'] / result['O3_20hour_last']
        result['30day_PM2.5_mean/60day_PM2.5_mean_rate'] = result['30day_PM2.5_mean_city'] / result['60day_PM2.5_mean_city']
        result['30day_PM10_mean/60day_PM10_mean_rate'] = result['30day_PM10_mean_city'] / result['60day_PM10_mean_city']
        result['30day_O3_mean/60day_O3_mean_rate'] = result['30day_O3_mean_city'] / result['60day_O3_mean_city']
        result['3day_PM2.5_mean/7day_PM2.5_mean_rate'] = result['3day_PM2.5_mean_city'] / result['7day_PM2.5_mean_city']
        result['3day_PM10_mean/7day_PM10_mean_rate'] = result['3day_PM10_mean_city'] / result['7day_PM10_mean_city']
        result['3day_O3_mean/7day_O3_mean_rate'] = result['3day_O3_mean_city'] / result['7day_O3_mean_city']
        result['1day_PM2.5_mean_city/2day_PM2.5_mean_city_rate'] = result['1day_PM2.5_mean_city'] / result['2day_PM2.5_mean_city']
        result['1day_PM10_mean_city/2day_PM10_mean_city_rate'] = result['1day_PM10_mean_city'] / result['2day_PM10_mean_city']
        result['1day_O3_mean_city/2day_O3_mean_city_rate'] = result['1day_O3_mean_city'] / result['2day_O3_mean_city']
        print('second feature computed...')
    except:
        pass
    return result

# compute feature for a particular date=data_key
def make_feat(data_key,silent=0,replace=False):
    print(end='') if silent else print('data key is：{}'.format(data_key))
    result_path = cache_path + 'feat_set_{}_{}hour_ago.hdf'.format(data_key,1)
    if os.path.exists(result_path) & (not replace):
        result = pd.read_hdf(result_path, 'w')
    else:
        data = pre_treatment(data_key)

#         convert dataframe type to list type
        result = [data]
        result.append(get_24hour_feat(data,data_key,replace))
#         choose the list of days compute properly
        for i in [1,2,3,7,15,30,60]:
#         for i in [1,2,3,7,15,30,60,360]:
            result.append(get_nday_mean_feat(data,data_key,i,replace))       
        result.append(get_weather_feat(data, data_key,replace))              

#         convert to dataframe
        result = concat(result)

        result = second_feat(result)
#         add labels
        result = get_label(result)
        result = convert_dtypes(result, result.columns, silent=True)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

