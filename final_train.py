
import pandas as pd
import numpy as np
import scipy as sp
import time,datetime
import lightgbm as lgb
import warnings,math
from scipy import stats
from sklearn import metrics,preprocessing
from sklearn.model_selection import KFold,train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def read_data(inputfile):
    names = [
        'uid','gender','age','education','territory_code','whether_to_keep','installation_channel','vid','classification',
        'tag','creator','upload_time','video_duration','show','click','recommended_type','video_playback_time','timestamp_of_behavior',
        'comment','like','forward'
    ]
    train_df =  pd.read_csv(inputfile, sep='\t',header=None,names=names)

    return train_df

# timestamp -> date
time_format = '%Y-%m-%d %H:%M:%S'
def get_date(timestamp) :
    time_local = time.localtime(timestamp)
    dt = time.strftime(time_format,time_local)
    
    return dt

def encode_count(df,column_name):
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df[column_name].values))
        df[column_name] = lbl.transform(list(df[column_name].values))
        
        return df

def preprocess_data(train_df):
    print('Deleting 删除territory code...')
    # 删除territory code 特征，因为整份数据都是相同的
    train_df.drop(['territory_code'],axis=1,inplace=True)

    print('Dividing 1000...')
    # 行为时间戳多了1000
    train_df['timestamp_of_behavior'] = train_df['timestamp_of_behavior'].apply(lambda row:row/1000)

    print('Sorting by time of behavior...')
    # 对数据按照uid和timestamp_of_behavior升序排序
    train_df.sort_values(['uid','timestamp_of_behavior'],ascending=True,inplace=True)
    train_df.reset_index(drop=True,inplace=True)
    
    # show这个特征比较奇特，-表示没有观看，而非缺失值
    train_df['show'] = train_df['show'].replace('-',0)

    print('Replace nan...')
    # 将缺失值‘-’替换为nan
    train_df = train_df.replace('-',np.nan)

    # 修改时间戳格式
    # 修改时间戳格式
    train_df['action_date'] = train_df['timestamp_of_behavior'].apply(get_date)
    train_df['action_date'] = pd.to_datetime(train_df['action_date'])

    train_df['upload_date'] = train_df['upload_time'].apply(get_date)
    train_df['upload_date'] = pd.to_datetime(train_df['upload_date'])

    train_df['action_day'] = train_df['action_date'].dt.day
    train_df['action_hour'] = train_df['action_date'].dt.hour
    train_df['action_minute'] = train_df['action_date'].dt.minute
    train_df['action_week'] = train_df['action_date'].dt.weekday

    train_df['upload_day'] = train_df['upload_date'].dt.day

    # 离散属性数字化(将缺失值也作为一种属性)
    train_df = encode_count(train_df,'gender')
    train_df = encode_count(train_df,'age')
    train_df = encode_count(train_df,'education')
    train_df = encode_count(train_df,'installation_channel')
    train_df = encode_count(train_df,'classification')
    train_df = encode_count(train_df,'recommended_type')

    # 处理性别不一致的情况,使用众数填充性别缺失值
    userid_gender_df = train_df[['uid', 'gender']].groupby('uid').agg(lambda x: stats.mode(x)[0][0]).reset_index()
    userid_gender_df.rename(columns={'gender': 'gender_new'}, inplace=True)

    train_df = pd.merge(train_df, userid_gender_df, on='uid', how='left')
    del train_df['gender']
    train_df.rename(columns={'gender_new': 'gender'}, inplace=True)

    del userid_gender_df

    # 转换数据类型
    feat_list = [
            'age', 'education', 'whether_to_keep', 'installation_channel','classification',
            'video_duration', 'show', 'click', 'recommended_type','comment', 'like','forward', 
            'action_day', 'action_hour', 'action_minute', 'action_week',
            'upload_day','gender'
    ]

    for feat in feat_list:
        train_df[feat] = train_df[train_df[feat].notnull()][feat].astype(np.int16)

    train_df['video_playback_time'] = train_df['video_playback_time'].astype(np.float16)
    
    return train_df

def merge_mean(df_1, df_2, columns, value, cname):
    add = pd.DataFrame(df_1[df_1[value].notnull()].groupby(columns)[value].mean()).reset_index()
    add.columns = columns + [cname]
    df_2 = df_2.merge(add,on=columns,how="left")
    
    return df_2

def merge_sum(df_1, df_2, columns, value, cname):
    add = pd.DataFrame(df_1[df_1[value].notnull()].groupby(columns)[value].sum()).reset_index()
    add.columns = columns + [cname]
    df_2 = df_2.merge(add,on=columns,how="left")
    
    return df_2

def merge_count(df_1, df_2, columns, value, cname):
    add = pd.DataFrame(df_1[df_1[value].notnull()].groupby(columns)[value].count()).reset_index()
    add.columns = columns + [cname]
    df_2 = df_2.merge(add,on=columns,how="left")
    
    return df_2

def merge_max(df_1, df_2, columns, value, cname):
    add = pd.DataFrame(df_1[df_1[value].notnull()].groupby(columns)[value].max()).reset_index()
    add.columns = columns + [cname]
    df_2 = df_2.merge(add,on=columns,how="left")
    
    return df_2

def merge_min(df_1, df_2, columns, value, cname):
    add = pd.DataFrame(df_1[df_1[value].notnull()].groupby(columns)[value].min()).reset_index()
    add.columns = columns + [cname]
    df_2 = df_2.merge(add,on=columns,how="left")
    
    return df_2

def merge_std(df_1, df_2, columns, value, cname):
    add = pd.DataFrame(df_1[df_1[value].notnull()].groupby(columns)[value].std()).reset_index()
    add.columns = columns + [cname]
    df_2 = df_2.merge(add,on=columns,how="left")
    
    return df_2

def count_self_merge(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].count()).reset_index()
    add.columns = columns + [cname]
    df = df.merge(add,on=columns,how="left")
    
    return df

def mean_self_merge(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].mean()).reset_index()
    add.columns = columns + [cname]
    df = df.merge(add,on=columns,how="left")
    
    return df

def std_self_merge(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].std()).reset_index()
    add.columns = columns + [cname]
    df = df.merge(add,on=columns,how="left")
    
    return df

def min_self_merge(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].min()).reset_index()
    add.columns = columns + [cname]
    df = df.merge(add,on=columns,how="left")
    
    return df

def max_self_merge(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].max()).reset_index()
    add.columns = columns + [cname]
    df = df.merge(add,on=columns,how="left")
    
    return df

# 用户所看tag数
def get_tag_nums(row):
    if row == '':
        return
    tag_list = row.split('$')
    
    return len(tag_list)-1

def get_tag_list(row):
    if row == '':
        return
    tag_list = row.split('$')
    tag_list.remove('')
    
    return tag_list

def get_all_tags_nums(df):
    all_tag_list = []
    all_tags = df[['tag_list']]
    
    for r in zip(all_tags['tag_list']):
        for tag in r[0]:
            all_tag_list.append(tag)
    return len(set(all_tag_list))

# 获取某用户的所有tag类别
def get_all_tag_list(row):
    tag_list = []
    
    for idx,val in row.items():
        if val == None:
            break
        for tag in val:
            tag_list.append(tag)
    
    return ' '.join(tag_list)

def max_recommond(row):
    alist = [row['recommended_type_0_counts'],row['recommended_type_1_counts'],row['recommended_type_2_counts']]
    return alist.index(max(alist)) 

def min_recommond(row):
    alist = [row['recommended_type_0_counts'],row['recommended_type_1_counts'],row['recommended_type_2_counts']]
    return alist.index(min(alist)) 

def feat_eng(train_df):
    # 最终数据
    print('Final data...')
    final_data_df = train_df[['uid','gender','age','education','whether_to_keep']].drop_duplicates(['uid'],keep='first').copy()
    print(final_data_df.shape)

    print('Concating cat feature...')

    print('Calculating timedelta...')
    # 行为时间和预测时间的差值
    train_df['action_time_delta'] = 31 - train_df['action_day']

    # 视频上传时间和预测时间的差值
    train_df['upload_time_delta'] = 31 - train_df['upload_day']

    # 观看时间和视频上传时间的差值(天)
    train_df['action_upload_time_delta'] = (train_df['action_date']-train_df['upload_date']).dt.total_seconds()/86400

    # 每个视频的被观看的时长和总长比例
    train_df['video_play_per'] = np.round(train_df['video_playback_time']/train_df['video_duration'],3).astype(np.float16)

    # 用户所有行为时间和预测时间的差值的平均值
    final_data_df = merge_mean(train_df,final_data_df,['uid'],'action_time_delta','action_time_delta_mean')
    # 用户所有行为时间和预测时间的差值的标准差
    final_data_df = merge_mean(train_df,final_data_df,['uid'],'action_time_delta','action_time_delta_std')

    # 用户所有视频上传时间和预测时间的差值的平均值
    final_data_df = merge_mean(train_df,final_data_df,['uid'],'upload_time_delta','upload_time_delta_mean')

    # 用户所有视频上传时间和预测时间的差值的标准差
    final_data_df = merge_std(train_df,final_data_df,['uid'],'upload_time_delta','upload_time_delta_std')

    # 用户所有观看时间和视频上传时间的差值的标准差
    final_data_df = merge_std(train_df,final_data_df,['uid'],'action_upload_time_delta','action_upload_time_delta_std')

    # 用户所有观看时间和视频上传时间的差值的均值
    final_data_df = merge_mean(train_df,final_data_df,['uid'],'action_upload_time_delta','action_upload_time_delta_mean')

    # 用户所有观看时间(剔除非0)和视频上传时间的差值的均值
    final_data_df = merge_mean(train_df[train_df['video_playback_time']!=0],final_data_df,['uid'],'action_upload_time_delta','action_upload_time_delta_mean_no_0')
    # 用户所有观看时间(剔除非0)和视频上传时间的差值的标准差
    final_data_df = merge_std(train_df[train_df['video_playback_time']!=0],final_data_df,['uid'],'action_upload_time_delta','action_upload_time_delta_std_no_0')

    # 用户所看的每个视频的被观看的时长和总长比例的均值
    final_data_df = merge_mean(train_df,final_data_df,['uid'],'video_play_per','video_play_per_mean')

    # 用户所看的每个视频的被观看的时长和总长比例的方差
    final_data_df = merge_std(train_df,final_data_df,['uid'],'video_play_per','video_play_per_std')

    # 用户所看的每个视频的被观看的时长和总长比例的均值（剔除观看时间0）
    final_data_df = merge_mean(train_df[train_df['video_playback_time']!=0],final_data_df,['uid'],'video_play_per','video_play_per_mean_no_0')

    # 用户所看的每个视频的被观看的时长和总长比例的方差（剔除观看时间0）
    final_data_df = merge_mean(train_df[train_df['video_playback_time']!=0],final_data_df,['uid'],'video_play_per','video_play_per_std_no_0')

    print('Calculating show/click(etc) times...')
    # 总的点赞评论转发数
    final_data_df = merge_sum(train_df,final_data_df,['uid'],'show','show_counts')
    final_data_df = merge_sum(train_df,final_data_df,['uid'],'like','like_counts')
    final_data_df = merge_sum(train_df,final_data_df,['uid'],'forward','forward_counts')
    final_data_df = merge_sum(train_df,final_data_df,['uid'],'comment','comment_counts')
    final_data_df = merge_sum(train_df,final_data_df,['uid'],'click','click_counts')

    # 用户所看所有视频的长度总和
    final_data_df = merge_sum(train_df,final_data_df,['uid'],'video_duration','video_duration_sum')

    # 用户所看所有视频的长度总和（剔除观看时间为0）
    final_data_df = merge_sum(train_df[train_df['video_playback_time']!=0],final_data_df,['uid'],'video_duration','video_duration_sum_no_0')

    # 用户所看所有时长的和
    final_data_df = merge_sum(train_df,final_data_df,['uid'],'video_playback_time','video_playback_time_sum')

    # 总体观看时间占总体视频长度和的比例
    final_data_df['video_play_sum_per'] = round(final_data_df['video_playback_time_sum']/final_data_df['video_duration_sum'],3).astype(np.float16)

    # 总体观看时间占总体视频长度(除去播放时间为0)和的比例
    final_data_df['valid_video_play_sum_per'] = round(final_data_df['video_duration_sum_no_0']/final_data_df['video_duration_sum'],3).astype(np.float16)

    print('Calculating all video/creator/install ways/class nums...')
    # 用户所看的所有视频数（含播放时间为0）
    final_data_df = merge_count(train_df,final_data_df,['uid'],'vid','video_nums_0')
    # 用户所看的所有视频数（bu含播放时间为0）
    final_data_df = merge_count(train_df[train_df['video_playback_time']!=0],final_data_df,['uid'],'vid','video_nums')

    # 用户所看的所有上传者数量（含播放时间为0）
    final_data_df = merge_count(train_df[['uid','creator']].drop_duplicates(),final_data_df,['uid'],'creator','creator_nums_0')
    # 用户所看的所有上传者数量（bu含播放时间为0）
    final_data_df = merge_count(train_df[train_df['video_playback_time']!=0][['uid','creator']].drop_duplicates(),final_data_df,['uid'],'creator','creator_nums')

    # 用户所看的所有类别数（含播放时间为0）
    final_data_df = merge_count(train_df[['uid','classification']].drop_duplicates(),final_data_df,['uid'],'classification','class_nums_0')
    # 用户所看的所有类别数（bu含播放时间为0）
    final_data_df = merge_count(train_df[train_df['video_playback_time']!=0][['uid','classification']].drop_duplicates(),final_data_df,['uid'],'classification','class_nums')

    # 用户所看的所有安装渠道数
    final_data_df = merge_count(train_df[['uid','installation_channel']].drop_duplicates(),final_data_df,['uid'],'installation_channel','install_nums')

    # 用户的一半观看次数(低于30%被判定为普通)
    final_data_df = merge_count(train_df[train_df['video_play_per']<=0.3],final_data_df,['uid'],'video_play_per','valid_play_count_30m')
    # 用户的有效观看次数(超过30%被判定为有效)
    final_data_df = merge_count(train_df[train_df['video_play_per']>0.3],final_data_df,['uid'],'video_play_per','valid_play_count_30')
    # 用户的密切观看次数(超过50%被判定为有效)
    final_data_df = merge_count(train_df[train_df['video_play_per']>0.5],final_data_df,['uid'],'video_play_per','valid_play_count_50')
    # 用户的忠实观看次数(超过70%被判定为忠实)
    final_data_df = merge_count(train_df[train_df['video_play_per']>0.7],final_data_df,['uid'],'video_play_per','valid_play_count_70')

    print('Getting feature importance and calculating imp')
    install_imp_df = pd.read_csv('./install_imp.csv',encoding='utf-8')
    user_install_imp_df = train_df.merge(install_imp_df,on='installation_channel',how='left')

    # 得到用户安装渠道的平均重要性
    final_data_df = merge_mean(user_install_imp_df,final_data_df,['uid'],'install_keep_proba','install_keep_proba_mean')
    del install_imp_df,user_install_imp_df

    action_hour_imp_df = pd.read_csv('./action_hour_imp_df.csv',encoding='utf-8')
    user_acthour_imp_df = train_df.merge(action_hour_imp_df,on='action_hour',how='left')

    # 得到用户action_hour平均重要性
    final_data_df = merge_mean(user_acthour_imp_df,final_data_df,['uid'],'action_hour_keep_proba','action_hour_keep_proba_mean')
    del action_hour_imp_df,user_acthour_imp_df

    action_day_imp_df = pd.read_csv('./action_day_imp_df.csv',encoding='utf-8')
    user_actday_imp_df = train_df.merge(action_day_imp_df,on='action_day',how='left')

    # 得到用户action_hday平均重要性
    final_data_df = merge_mean(user_actday_imp_df,final_data_df,['uid'],'action_day_keep_proba','action_day_keep_proba_mean')
    del action_day_imp_df,user_actday_imp_df

    class_imp_df = pd.read_csv('./class_imp_df.csv',encoding='utf-8')
    user_class_imp_df = train_df.merge(class_imp_df,on='classification',how='left')

    # 得到用户classification平均重要性
    final_data_df = merge_mean(user_class_imp_df,final_data_df,['uid'],'class_keep_proba','class_keep_proba_mean')
    del user_class_imp_df,class_imp_df

    education_imp_df = pd.read_csv('./education_imp_df.csv',encoding='utf-8')
    user_education_imp_df = train_df.merge(education_imp_df,on='education',how='left')

    # 得到用户education平均重要性
    final_data_df = merge_mean(user_education_imp_df,final_data_df,['uid'],'education_keep_proba','education_keep_proba_mean')
    del user_education_imp_df,education_imp_df

    age_imp_df = pd.read_csv('./age_imp_df.csv',encoding='utf-8')
    user_age_imp_df = train_df.merge(age_imp_df,on='age',how='left')

    # 得到用户age平均重要性
    final_data_df = merge_mean(user_age_imp_df,final_data_df,['uid'],'age_keep_proba','age_keep_proba_mean')
    del user_age_imp_df,age_imp_df

    # 这个用户show总次数，占总观看次数的比例
    final_data_df['show_videonum0_per'] = round(final_data_df['show_counts']/final_data_df['video_nums'],3).astype(np.float16)
    # 这个用户show总次数，占总观看次数(去除播放时间为0)的比例
    final_data_df['show_videonum_per'] = round(final_data_df['show_counts']/final_data_df['video_nums_0'],3).astype(np.float16)

    # 这个用户click总次数，占总观看次数的比例
    final_data_df['click_videonum0_per'] = round(final_data_df['click_counts']/final_data_df['video_nums'],3).astype(np.float16)
    # 这个用户click总次数，占总观看次数(去除播放时间为0)的比例
    final_data_df['click_videonum_per'] = round(final_data_df['click_counts']/final_data_df['video_nums_0'],3).astype(np.float16)

    # 这个用户like总次数，占总观看次数的比例
    final_data_df['like_videonum0_per'] = round(final_data_df['like_counts']/final_data_df['video_nums'],3).astype(np.float16)
    # 这个用户click总次数，占总观看次数(去除播放时间为0)的比例
    final_data_df['like_videonum_per'] = round(final_data_df['like_counts']/final_data_df['video_nums_0'],3).astype(np.float16)

    # # 这个用户forward总次数，占总观看次数的比例
    final_data_df['forward_videonum0_per'] = round(final_data_df['forward_counts']/final_data_df['video_nums'],3).astype(np.float16)
    # 这个用户click总次数，占总观看次数(去除播放时间为0)的比例
    final_data_df['forward_videonum_per'] = round(final_data_df['forward_counts']/final_data_df['video_nums_0'],3).astype(np.float16)
    
    # 观看比例小于30%的show次数,与其占有效播放的占比
    final_data_df = merge_sum(train_df[train_df['video_play_per']<=0.3],final_data_df,['uid'],'show','show_30m_count')
    final_data_df['show_30m_count_per'] = round(final_data_df['show_30m_count']/final_data_df['valid_play_count_30m'],3).astype(np.float16)
    # 观看比例超过30%的show次数
    final_data_df = merge_sum(train_df[train_df['video_play_per']>0.3],final_data_df,['uid'],'show','show_30_count')
    final_data_df['show_30_count_per'] = round(final_data_df['show_30_count']/final_data_df['valid_play_count_30'],3).astype(np.float16)
    # 观看比例超过50%的show次数
    final_data_df = merge_sum(train_df[train_df['video_play_per']>0.5],final_data_df,['uid'],'show','show_50_count')
    final_data_df['show_50_count_per'] = round(final_data_df['show_50_count']/final_data_df['valid_play_count_50'],3).astype(np.float16)
    # 观看比例超过70%的show次数
    final_data_df = merge_sum(train_df[train_df['video_play_per']>0.7],final_data_df,['uid'],'show','show_70_count')
    final_data_df['show_70_count_per'] = round(final_data_df['show_70_count']/final_data_df['valid_play_count_70'],3).astype(np.float16)

    # 观看比例小于30%的click次数,与其占有效播放的占比
    final_data_df = merge_sum(train_df[train_df['video_play_per']<=0.3],final_data_df,['uid'],'click','click_30m_count')
    final_data_df['click_30m_count_per'] = round(final_data_df['click_30m_count']/final_data_df['valid_play_count_30m'],3).astype(np.float16)
    # 观看比例超过30%的click次数
    final_data_df = merge_sum(train_df[train_df['video_play_per']>0.3],final_data_df,['uid'],'click','click_30_count')
    final_data_df['click_30_count_per'] = round(final_data_df['click_30_count']/final_data_df['valid_play_count_30'],3).astype(np.float16)
    # 观看比例超过50%的click次数
    final_data_df = merge_sum(train_df[train_df['video_play_per']>0.5],final_data_df,['uid'],'click','click_50_count')
    final_data_df['click_50_count_per'] = round(final_data_df['click_50_count']/final_data_df['valid_play_count_50'],3).astype(np.float16)
    # 观看比例超过70%的click次数
    final_data_df = merge_sum(train_df[train_df['video_play_per']>0.7],final_data_df,['uid'],'click','click_70_count')
    final_data_df['click_70_count_per'] = round(final_data_df['click_70_count']/final_data_df['valid_play_count_70'],3).astype(np.float16)

    # 观看比例小于30%的like次数,与其占有效播放的占比
    final_data_df = merge_sum(train_df[train_df['video_play_per']<=0.3],final_data_df,['uid'],'like','like_30m_count')
    final_data_df['like_30m_count_per'] = round(final_data_df['like_30m_count']/final_data_df['valid_play_count_30m'],3).astype(np.float16)
    # 观看比例超过30%的like次数
    final_data_df = merge_sum(train_df[train_df['video_play_per']>0.3],final_data_df,['uid'],'like','like_30_count')
    final_data_df['like_30_count_per'] = round(final_data_df['like_30_count']/final_data_df['valid_play_count_30'],3).astype(np.float16)
    # 观看比例超过50%的like次数
    final_data_df = merge_sum(train_df[train_df['video_play_per']>0.5],final_data_df,['uid'],'like','like_50_count')
    final_data_df['like_50_count_per'] = round(final_data_df['like_50_count']/final_data_df['valid_play_count_50'],3).astype(np.float16)
    # 观看比例超过70%的like次数
    final_data_df = merge_sum(train_df[train_df['video_play_per']>0.7],final_data_df,['uid'],'like','like_70_count')
    final_data_df['like_70_count_per'] = round(final_data_df['like_70_count']/final_data_df['valid_play_count_70'],3).astype(np.float16)
    
    # 观看比例小于30%的comment次数,与其占有效播放的占比
    final_data_df = merge_sum(train_df[train_df['video_play_per']<=0.3],final_data_df,['uid'],'comment','comment_30m_count')
    final_data_df['comment_30m_count_per'] = round(final_data_df['comment_30m_count']/final_data_df['valid_play_count_30m'],3).astype(np.float16)
    # 观看比例超过30%的comment次数
    final_data_df = merge_sum(train_df[train_df['video_play_per']>0.3],final_data_df,['uid'],'comment','comment_30_count')
    final_data_df['comment_30_count_per'] = round(final_data_df['comment_30_count']/final_data_df['valid_play_count_30'],3).astype(np.float16)
    # 观看比例超过50%的comment次数
    final_data_df = merge_sum(train_df[train_df['video_play_per']>0.5],final_data_df,['uid'],'comment','comment_50_count')
    final_data_df['comment_50_count_per'] = round(final_data_df['comment_50_count']/final_data_df['valid_play_count_50'],3).astype(np.float16)
    # 观看比例超过70%的comment次数
    final_data_df = merge_sum(train_df[train_df['video_play_per']>0.7],final_data_df,['uid'],'comment','comment_70_count')
    final_data_df['comment_70_count_per'] = round(final_data_df['comment_70_count']/final_data_df['valid_play_count_70'],3).astype(np.float16)

    # 观看比例小于30%的forward次数,与其占有效播放的占比
    final_data_df = merge_sum(train_df[train_df['video_play_per']<=0.3],final_data_df,['uid'],'forward','forward_30m_count')
    final_data_df['forward_30m_count_per'] = round(final_data_df['forward_30m_count']/final_data_df['valid_play_count_30m'],3).astype(np.float16)
    # 观看比例超过30%的forward次数
    final_data_df = merge_sum(train_df[train_df['video_play_per']>0.3],final_data_df,['uid'],'forward','forward_30_count')
    final_data_df['forward_30_count_per'] = round(final_data_df['forward_30_count']/final_data_df['valid_play_count_30'],3).astype(np.float16)
    # 观看比例超过50%的forward次数
    final_data_df = merge_sum(train_df[train_df['video_play_per']>0.5],final_data_df,['uid'],'forward','forward_50_count')
    final_data_df['forward_50_count_per'] = round(final_data_df['forward_50_count']/final_data_df['valid_play_count_50'],3).astype(np.float16)
    # 观看比例超过70%的forward次数
    final_data_df = merge_sum(train_df[train_df['video_play_per']>0.7],final_data_df,['uid'],'forward','forward_70_count')
    final_data_df['forward_70_count_per'] = round(final_data_df['forward_70_count']/final_data_df['valid_play_count_70'],3).astype(np.float16)

    print('Processing tag...')

    # 这个用户所看的这个视频的标签数
    train_df['tag'] = train_df['tag'].fillna('')
    train_df['tag_nums'] = train_df['tag'].apply(get_tag_nums)
    train_df['tag_list'] = train_df['tag'].apply(get_tag_list)

    # 这个用户所看的所有标签的数量
    tag_groupd = train_df[train_df['tag']!=''].groupby(['uid']).apply(get_all_tags_nums).reset_index()
    tag_groupd.rename(columns={0:'all_tag_counts'},inplace=True)
    final_data_df = pd.merge(final_data_df,tag_groupd,on='uid',how='left')
    del tag_groupd

    # 这个用户所看的平均标签数(含播放时间为0)
    final_data_df['mean_tag_0'] = round(final_data_df['all_tag_counts']/final_data_df['video_nums_0'],3).astype(np.float16)
    # 这个用户所看的平均标签数(不含播放时间为0)
    final_data_df['mean_tag'] = round(final_data_df['all_tag_counts']/final_data_df['video_nums'],3).astype(np.float16)

    # # 同一userid下播放时长为0的次数
    # # 播放的时长为0的视频占总视频数的比例
    final_data_df = merge_count(train_df[train_df['video_playback_time']==0],final_data_df,['uid'],'video_playback_time','play_0_video_nums')
    final_data_df['play_0_video_nums_per'] = round(final_data_df['play_0_video_nums']/final_data_df['video_nums_0'],3).astype(np.float16)

    print('Getting the earlist and latest time of a user...')
    # 用户的最早有效（不含0）播放时间（小时）
    final_data_df = merge_min(train_df[train_df['video_playback_time']!=0],final_data_df,['uid'],'action_hour','min_valid_play_hour')
    # 用户的最早（含0）播放时间（小时）
    final_data_df = merge_min(train_df,final_data_df,['uid'],'action_hour','min_play_hour')
    # 用户的最早（含0）播放时间（分钟）
    final_data_df = merge_min(train_df,final_data_df,['uid'],'action_minute','min_play_min')
    # # 用户的最早（含0）播放时间（秒）
    # final_data_df = merge_min(train_df,final_data_df,['uid'],'action_second','min_play_second')

    # 用户的最晚有效（不含0）播放时间（小时）
    final_data_df = merge_max(train_df[train_df['video_playback_time']!=0],final_data_df,['uid'],'action_hour','max_valid_play_hour')
    # 用户的最晚（含0）播放时间（小时）
    final_data_df = merge_max(train_df,final_data_df,['uid'],'action_hour','max_play_hour')
    # 用户的最晚（含0）播放时间（分钟）
    final_data_df = merge_max(train_df,final_data_df,['uid'],'action_minute','max_play_min')
    # # 用户的最晚（含0）播放时间（秒）
    # final_data_df = merge_max(train_df,final_data_df,['uid'],'action_second','max_play_second')

    # 用户行为日期
    user_act = train_df[['uid','action_day','action_week','action_time_delta']].drop_duplicates(['uid']).copy()

    # 行为日期是否为预测期前1，3，5，7
    final_data_df = final_data_df.merge(user_act,on='uid',how='left')
    del user_act

    final_data_df['before_1_day'] = final_data_df['action_time_delta'].apply(lambda x:1 if x<=1 else 0)
    final_data_df['before_3_day'] = final_data_df['action_time_delta'].apply(lambda x:1 if x<=3 else 0)
    final_data_df['before_5_day'] = final_data_df['action_time_delta'].apply(lambda x:1 if x<=5 else 0)
    final_data_df['before_7_day'] = final_data_df['action_time_delta'].apply(lambda x:1 if x<=7 else 0)

    # 用户有效播放时间差
    final_data_df['valid_hour_delta'] = (final_data_df['max_valid_play_hour']-final_data_df['min_valid_play_hour']).astype(np.float16)

    # 用户播放时间差
    final_data_df['hour_delta'] = (final_data_df['max_play_hour']-final_data_df['min_play_hour']).astype(np.float16)

    # 用户和视频交互行为总和（comment，forward，like）
    final_data_df['comm_forw_like_sum'] = (final_data_df['comment_counts']+final_data_df['forward_counts']+final_data_df['like_counts']).astype(np.float16)
    # 用户show+click次数
    final_data_df['show_click_sum'] = (final_data_df['show_counts']+final_data_df['click_counts']).astype(np.float16)
    # 用户所有行为的和
    final_data_df['all_behave_sum'] = (final_data_df['comm_forw_like_sum']+final_data_df['show_click_sum']).astype(np.float16)

    print('Cal the gap between two videos...')
    # 用户浏览两条视频的间隔（s）
    train_df['action_time_diff'] = train_df.groupby('uid')['timestamp_of_behavior'].apply(lambda i:i.diff(1))

    # 间隔的最大值
    final_data_df = merge_max(train_df,final_data_df,['uid'],'action_time_diff','action_time_diff_max')
    # 间隔的平均值
    final_data_df = merge_mean(train_df,final_data_df,['uid'],'action_time_diff','action_time_diff_mean')
    # 间隔的方差
    final_data_df = merge_std(train_df,final_data_df,['uid'],'action_time_diff','action_time_diff_std')

    # 用户浏览两条视频的有效间隔（s）
    # 用户-行为时间（视频播放时间不为0）
    user_valid_action = train_df[train_df['video_playback_time']!=0][['uid','timestamp_of_behavior']].copy()
    user_valid_action['valid_action_time_diff'] = user_valid_action.groupby('uid')['timestamp_of_behavior'].apply(lambda i:i.diff(1))

    # 间隔的最大值
    final_data_df = merge_max(user_valid_action,final_data_df,['uid'],'valid_action_time_diff','valid_action_time_diff_max')
    # 间隔的最小值
    final_data_df = merge_min(user_valid_action,final_data_df,['uid'],'valid_action_time_diff','valid_action_time_diff_min')
    # 间隔的平均值
    final_data_df = merge_mean(user_valid_action,final_data_df,['uid'],'valid_action_time_diff','valid_action_time_diff_mean')
    # 间隔的方差
    final_data_df = merge_std(user_valid_action,final_data_df,['uid'],'valid_action_time_diff','valid_action_time_diff_std')

    del user_valid_action

    # 用户所有行为的和占总观看次数的比例
    final_data_df['all_behave_videonum0_per'] = round(final_data_df['all_behave_sum']/final_data_df['video_nums'],3).astype(np.float16)
    # 用户所有行为的和占总观看次数的比例(去除播放时间为0)
    final_data_df['all_behave_videonum_per'] = round(final_data_df['all_behave_sum']/final_data_df['video_nums_0'],3).astype(np.float16)

    # 观看比例小于30%的所有行为次数次数,与其占有效播放的占比
    final_data_df['all_behave_sum_30m'] = final_data_df['show_30m_count'] + final_data_df['click_30m_count'] + final_data_df['like_30m_count'] + final_data_df['comment_30m_count'] + final_data_df['forward_30m_count']
    final_data_df['all_behave_sum_30m_per'] = round(final_data_df['all_behave_sum_30m']/final_data_df['valid_play_count_30m'],3).astype(np.float16)

    # 观看比例大于30%的所有行为次数次数,与其占有效播放的占比
    final_data_df['all_behave_sum_30'] = final_data_df['show_30_count'] + final_data_df['click_30_count'] + final_data_df['like_30_count'] + final_data_df['comment_30_count'] + final_data_df['forward_30_count']
    final_data_df['all_behave_sum_30_per'] = round(final_data_df['all_behave_sum_30']/final_data_df['valid_play_count_30'],3).astype(np.float16)

    # 观看比例大于50%的所有行为次数次数,与其占有效播放的占比
    final_data_df['all_behave_sum_50'] = final_data_df['show_50_count'] + final_data_df['click_50_count'] + final_data_df['like_50_count'] + final_data_df['comment_50_count'] + final_data_df['forward_50_count']
    final_data_df['all_behave_sum_50_per'] = round(final_data_df['show_50_count']/final_data_df['valid_play_count_50'],3).astype(np.float16)

    # 观看比例大于70%的所有行为次数次数,与其占有效播放的占比
    final_data_df['all_behave_sum_70'] = final_data_df['show_70_count'] + final_data_df['click_70_count'] + final_data_df['like_70_count'] + final_data_df['comment_70_count'] + final_data_df['forward_70_count']
    final_data_df['all_behave_sum_70_per'] = round(final_data_df['show_70_count']/final_data_df['valid_play_count_70'],3).astype(np.float16)

    print('Process recommond feature...')
    # 每个用户总共的推荐类型次数
    uid_recommond_uique_df = train_df[['uid','recommended_type']].drop_duplicates().copy()
    final_data_df = merge_count(uid_recommond_uique_df,final_data_df,['uid'],'recommended_type','recommended_type_counts')
    del uid_recommond_uique_df

    # 每个用户总共的有效推荐类型次数
    valid_uid_recommond_uique_df = train_df[train_df['video_playback_time']!=0][['uid','recommended_type']].drop_duplicates(['uid','recommended_type']).copy()
    final_data_df = merge_count(valid_uid_recommond_uique_df,final_data_df,['uid'],'recommended_type','valid_recommended_type_counts')
    del valid_uid_recommond_uique_df

    # 推荐成功比例
    final_data_df['recommond_sucess_per'] = round(final_data_df['valid_recommended_type_counts']/final_data_df['recommended_type_counts'],3).astype(np.float16)

    # 每个用户，被推荐的每个类型的个数
    final_data_df = merge_count(train_df[train_df['recommended_type']==0],final_data_df,['uid'],'action_day','recommended_type_0_counts')
    final_data_df = merge_count(train_df[train_df['recommended_type']==1],final_data_df,['uid'],'action_day','recommended_type_1_counts')
    final_data_df = merge_count(train_df[train_df['recommended_type']==2],final_data_df,['uid'],'action_day','recommended_type_2_counts')
        
    final_data_df['max_recommond_type'] = final_data_df.apply(max_recommond,axis=1)
    final_data_df['min_recommond_type'] = final_data_df.apply(min_recommond,axis=1)

    print('Hot creator feature...')
    # 构建网红视频博主
    # 统计ceator上传了多少视频
    train_df = merge_count(train_df[train_df['creator'].notnull()].drop_duplicates(['creator','vid']),train_df,['creator'],'vid','video_count_creator')

    # 统计这个creator被多少用户观看
    train_df = merge_count(train_df[(train_df['creator'].notnull())].drop_duplicates(['uid','creator']),train_df,['creator'],'uid','creator_fans_num')

    # 统计这个creator发布的所有视频被观看的次数
    train_df = merge_count(train_df[(train_df['creator'].notnull())],train_df,['creator'],'uid','creator_play_times')

    creator_df = train_df[['creator','video_count_creator','creator_fans_num','creator_play_times']].drop_duplicates()

    v1 = creator_df.describe()['video_count_creator']['mean']
    v2 = creator_df.describe()['creator_fans_num']['mean']
    v3 = creator_df.describe()['creator_play_times']['mean']

    def whether_hot(condition_1,condition_2,condition_3):
        if((condition_1>=v1)&(condition_2>=v2)&(condition_3>=v3)):
            return 1
        else:
            return 0
        
    creator_df['whether_hot_creator'] = list(map(whether_hot,creator_df['video_count_creator'],creator_df['creator_fans_num'],creator_df['creator_play_times']))
    creator_df.reset_index(drop=True,inplace=True)

    train_df = train_df.merge(creator_df[['creator','whether_hot_creator']],on='creator',how='left')
    del creator_df

    # 用户观看网红creator的个数
    final_data_df = merge_sum(train_df[['uid','creator','whether_hot_creator']].drop_duplicates(),final_data_df,['uid'],'whether_hot_creator','hot_creator_nums')

    # 用户观看网红creator的占比
    final_data_df['hot_creator_nums_per'] = round(final_data_df['hot_creator_nums']/final_data_df['creator_nums_0'],3).astype(np.float16)

    print('Hot videos feature...')
    # 统计用户观看热门视频的次数
    # 统计每个vid被观看的次数
    train_df = count_self_merge(train_df,['vid'] ,'uid','video_play_counts')

    # 统计每个vid被观看的人数
    train_df = merge_count(train_df.drop_duplicates(['uid','vid']),train_df,['vid'],'uid','video_play_people_nums')

    # 统计每个vid被观看的平均比例
    train_df = mean_self_merge(train_df,['vid'],'video_play_per','one_video_play_per')

    video_df = train_df[['vid','video_play_counts','video_play_people_nums','one_video_play_per']].drop_duplicates().copy()

    v1 = video_df.describe()['video_play_counts']['mean']
    v2 = video_df.describe()['video_play_people_nums']['mean']
    v3 = video_df.describe()['one_video_play_per']['mean']

    def whether_hot(condition_1,condition_2,condition_3):
        if((condition_1>=v1)&(condition_2>=v2)&(condition_3>=v3)):
            return 1
        else:
            return 0
        
    video_df['whether_hot_video'] = list(map(whether_hot,video_df['video_play_people_nums'],video_df['video_play_people_nums'],video_df['one_video_play_per']))
    video_df.reset_index(drop=True,inplace=True)

    train_df = train_df.merge(video_df[['vid','whether_hot_video']],on='vid',how='left')
    del video_df

    # 用户观看热门video的个数
    final_data_df = merge_count(train_df[['uid','vid','whether_hot_video']],final_data_df,['uid'],'whether_hot_video','hot_video_nums')

    # 用户观看热门video的占比
    final_data_df['hot_video_nums_per'] = round(final_data_df['hot_video_nums']/final_data_df['video_nums_0'],3).astype(np.float16)

    # 用户的平均观看时长
    final_data_df['video_play_mean'] = round(final_data_df['video_playback_time_sum']/final_data_df['video_nums'],3).astype(np.float16)

    # 用户所看视频本身的平均时长('看了'means 不能包含播放时长为0的情况)
    final_data_df['video_duration_mean'] = round(final_data_df['video_duration_sum_no_0']/final_data_df['video_nums'],3).astype(np.float16)

    print('Before...')
    # 是否在预测期的前1h，2h，3h内有行为
    final_data_df['whether_before_1h'] = final_data_df[final_data_df['action_day']==31]['max_play_hour'].apply(lambda x:1 if x==23 else 0)
    final_data_df['whether_before_3h'] = final_data_df[final_data_df['action_day']==31]['max_play_hour'].apply(lambda x:1 if(x==22 or x==21) else 0)

    # 行为时间是否是周末
    final_data_df['is_weekend'] = final_data_df['action_week'].apply(lambda x:1 if (x==5 or x==6) else 0)

    print('Via upload time predict whether the video is great...')
    # 视频上传时间距现在的时间差，时间差越大，说明这个视频更加精品
    # 视频的平均观看-上传时间差分箱
    old_hot_video = train_df[['vid','action_upload_time_delta']].groupby(['vid'])['action_upload_time_delta'].mean()

    bins = [0,old_hot_video.describe()['25%'],old_hot_video.describe()['50%'],old_hot_video.describe()['75%'],old_hot_video.describe()['max']]
    labels = [0,1,2,3]

    delta_imp_df = pd.cut(old_hot_video,bins,labels=labels).astype(np.float16).reset_index()
    delta_imp_df.rename(columns={'action_upload_time_delta':'action_upload_time_delta_imp'},inplace=True)

    train_df = train_df.merge(delta_imp_df,on='vid',how='left')
    del old_hot_video,delta_imp_df

    train_df = mean_self_merge(train_df,['uid'],'action_upload_time_delta_imp','action_upload_time_delta_imp_mean')

    # 用户所看视频的精品程度
    final_data_df = final_data_df.merge(train_df[['uid','action_upload_time_delta_imp_mean']].drop_duplicates(),on='uid',how='inner')

    print('User & behaviors...')
    # 年龄和刷视频数量的关系
    age_behavior_df = final_data_df[['age','video_nums_0']].groupby('age').mean().reset_index()
    age_behavior_df.columns = ['age','age_video_nums_imp']

    final_data_df = final_data_df.merge(age_behavior_df,on='age',how='left')

    # 性别和刷视频数量的关系
    gender_behavior_df = final_data_df[['gender','video_nums_0']].groupby('gender').mean().reset_index()
    gender_behavior_df.columns = ['gender','gender_video_nums_imp']

    final_data_df = final_data_df.merge(gender_behavior_df,on='gender',how='left')

    # 学历和刷视频的数量的关系
    edu_behavior_df = final_data_df[['education','video_nums_0']].groupby('education').mean().reset_index()
    edu_behavior_df.columns = ['education','edu_video_nums_imp']

    final_data_df = final_data_df.merge(edu_behavior_df,on='education',how='left')

    del age_behavior_df,gender_behavior_df,edu_behavior_df

    print('User & time delta...')
    # 年龄与行为时间差
    age_behavior_df = final_data_df[['age','hour_delta']].groupby('age').mean().reset_index()
    age_behavior_df.columns = ['age','age_hour_delta_imp']

    final_data_df = final_data_df.merge(age_behavior_df,on='age',how='left')

    # 性别和行为时间差
    gender_behavior_df = final_data_df[['gender','hour_delta']].groupby('gender').mean().reset_index()
    gender_behavior_df.columns = ['gender','gender_hour_delta_imp']

    final_data_df = final_data_df.merge(gender_behavior_df,on='gender',how='left')

    # 学历和行为时间差
    edu_behavior_df = final_data_df[['education','hour_delta']].groupby('education').mean().reset_index()
    edu_behavior_df.columns = ['education','edu_hour_delta_imp']

    final_data_df = final_data_df.merge(edu_behavior_df,on='education',how='left')

    del age_behavior_df,gender_behavior_df,edu_behavior_df

    # 及时删除释放内存
    del train_df

    # 保存下来以免后面报错
    final_data_df.to_csv('./final_data.csv',encoding='utf-8')
    print(final_data_df.shape)
    return final_data_df

def train_model(final_data_df,model_path):
    # 获取所有特征列
    print('Getting all the feature...')
    lgb_feature_list = list(final_data_df.columns.drop(['uid']))
    lgb_df = final_data_df[lgb_feature_list].copy()
    target = 'whether_to_keep'

    # 划分数据
    print('Dividing dataset to trainset and valset...')
    train,val = train_test_split(lgb_df,test_size=0.2,random_state=2018)

    train_X = train.drop(target,1)
    train_y = train[target]

    val_X = val.drop(target,1)
    val_y = val[target]

    # 及时删除释放内存
    del final_data_df

    feature_name = lgb_feature_list.remove(target)

    lgb_train = lgb.Dataset(train_X,train_y,feature_name=feature_name)
    lgb_eval = lgb.Dataset(val_X, val_y, feature_name=feature_name,reference=lgb_train)

    # 保存 Dataset 到 LightGBM 二进制文件将会使得加载更快速:
    print('Saving trainset and valset...')
    lgb_train.save_binary('./train.bin')
    lgb_eval.save_binary('./val.bin')

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_thread': -1,
        'num_leaves':70,
        'max_depth':7,
        'learning_rate':0.01,
        'bagging_freq': 4,
        'bagging_fraction': 0.6,
        'feature_fraction': 0.6,
        'lambda_l1':1,
        'lambda_l2':1,
        'num_boost_round':20000,
        'data_random_seed':2017
    }

    ## 训练
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=lgb_eval,
        early_stopping_rounds=100
    )
    ### 保存模型
    model.save_model(model_path,num_iteration=model.best_iteration)

    # 保存模型重要性
    importance = model.feature_importance()
    names = model.feature_name()
    with open('./feature_importance.txt', 'w+') as file:
        for index, im in enumerate(importance):
            string = names[index] + ', ' + str(im) + '\n'
            file.write(string)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print('Reading data...')
    inputfile = './demo.txt'
    train_df = read_data(inputfile)

    print('Preprocessing data...')
    train_df = preprocess_data(train_df)

    print('Feature Engineer...')
    final_data_df = feat_eng(train_df)

    # print('Trainning LGB model...')
    # model_path = './model.txt'
    # train_model(final_data_df,model_path)
    
    print('Finishing....')
