import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import copy
import gc
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import json
import random
random.seed(4)


data_folder = './data/'
subm_folder = './submissions/'

# считываем исходные данные
df_train = pd.read_csv(os.path.join(data_folder, 'onetwotrip_challenge_train.csv'))
df_test = pd.read_csv(os.path.join(data_folder,'onetwotrip_challenge_test.csv'))

# переводим хэшированные userid в int
all_user_ids = list(df_train["userid"].unique()) + list(df_test["userid"].unique())
all_user_ids_dict = dict(zip(all_user_ids, range(len(all_user_ids))))

df_train['userid'] = df_train['userid'].map(all_user_ids_dict)
df_test['userid'] = df_test['userid'].map(all_user_ids_dict)

# отстортировываем данные по времени внутри каждого юзера (field4 - номер заказа юзера)
df_train.sort_values(['userid', 'field4'], inplace=True)
df_test.sort_values(['userid', 'field4'], inplace=True)

# небольшое преобразование field1 и field14
# посокольку эти признаки нормализованы, была предпринята попытка вернуть им исходный вид
df_train['field1_1'] = round(df_train['field1']*6445.62 + 8538,-2)
df_test['field1_1'] = round(df_test['field1']*6445.62 + 8538,-2)

df_train['field14_1'] = round(df_train['field14']*357 + 2686,-2)
df_test['field14_1'] = round(df_test['field14']*357 + 2686,-2)


# создание признака часов до вылета (field16 - количество дней до вылета с момента покупки)
def get_hours(field16, hour_buy, hour_flight):
    if field16==0:
        return hour_flight-hour_buy
    else:
        return (24-hour_buy) + hour_flight +(field16-1)*24

df_train['hours_to_flight'] = df_train.apply(lambda x: get_hours(x['field16'], x['field11'], x['field23']), axis=1) 
df_test['hours_to_flight'] = df_test.apply(lambda x: get_hours(x['field16'], x['field11'], x['field23']), axis=1)

# field0 - дней с предыдущей покупки. Заменим 0 на None для первичников в этом признаке
df_train.loc[df_train.field5==1, 'field0'] = np.nan
df_test.loc[df_test.field5==1, 'field0'] = np.nan


fields_cols = list(filter(lambda x: 'field' in x, df_train.columns))
ind_cols = list(filter(lambda x: 'indicator' in x, df_train.columns))
cols_for_shift = fields_cols + ind_cols

# смотрим какие признаки были в предыдущей и следующей покупке
name_cols_shift1 = ['shift1_'+col for col in cols_for_shift]
name_cols_shiftminus = ['shiftminus_'+col for col in cols_for_shift]

df_train[name_cols_shift1] = df_train.groupby('userid')[cols_for_shift].shift(1)
df_test[name_cols_shift1] = df_test.groupby('userid')[cols_for_shift].shift(1)
for col, shift_col in zip(cols_for_shift, name_cols_shift1):
    df_train[col+'_dif'] = df_train[col] - df_train[shift_col]
    df_test[col+'_dif'] = df_test[col] - df_test[shift_col]

df_train[name_cols_shiftminus] = df_train.groupby('userid')[cols_for_shift].shift(-1)
df_test[name_cols_shiftminus] = df_test.groupby('userid')[cols_for_shift].shift(-1)
for col, shift_col in zip(cols_for_shift, name_cols_shiftminus):
    df_train[col+'_minusdif'] = df_train[col] - df_train[shift_col]
    df_test[col+'_minusdif'] = df_test[col] - df_test[shift_col]


# mean encoding всех goal по field12
df_train['field12_cut'] = pd.cut(df_train['field12'], [0.999,2,3,4,5,8,9,15,25,500])
df_test['field12_cut'] = pd.cut(df_test['field12'], [0.999,2,3,4,5,8,9,15,25,500])
for targ in ['goal21', 'goal22', 'goal23', 'goal24', 'goal25', 'goal1']:
    map_dict = dict(df_train.groupby('field12_cut')[targ].mean())

    df_train['field12_mean_encoding_'+targ] = df_train['field12_cut'].map(map_dict).astype(np.float64)
    df_test['field12_mean_encoding_'+targ] = df_test['field12_cut'].map(map_dict).astype(np.float64)

df_train.drop('field12_cut', axis=1, inplace=True)
df_test.drop('field12_cut', axis=1, inplace=True)

# признак - когда в последний раз field6 был не нулем
df_train['field6_mod'] = df_train['shift1_field6'].apply(lambda x: 1 if x >0 else x)
df_train['field6_strange_feature'] = (df_train['field6_mod']* df_train['field4']-1).fillna(0).replace({-1:None}).fillna(method='ffill')
df_train.drop(['field6_mod'], axis=1)
df_train['field6_strange_feature_dif'] = df_train['field6_mod'] - df_train['field6'] 

df_test['field6_mod'] = df_test['shift1_field6'].apply(lambda x: 1 if x >0 else x)
df_test['field6_strange_feature'] = (df_test['field6_mod']* df_test['field4']-1).fillna(0).replace({-1:None}).fillna(method='ffill')
df_test.drop(['field6_mod'], axis=1)
df_test['field6_strange_feature_dif'] = df_test['field6_mod'] - df_test['field6'] 

# ??? (не знаю, что я сделал и зачем, но этот признак есть в итоговм решении:)
df_train['field0_mod'] = df_train['shift1_field0'].apply(lambda x: 1 if x >0 else x)
df_train['field0_strange_feature'] = (df_train['field0_mod']* df_train['field4']-1).fillna(0).replace({-1:None}).fillna(method='ffill')
df_train.drop(['field0_mod'], axis=1)
df_train['field0_strange_feature_dif'] = df_train['field0_mod'] - df_train['field0'] 

df_test['field0_mod'] = df_test['shift1_field0'].apply(lambda x: 1 if x >0 else x)
df_test['field0_strange_feature'] = (df_test['field0_mod']* df_test['field4']-1).fillna(0).replace({-1:None}).fillna(method='ffill')
df_test.drop(['field0_mod'], axis=1)
df_test['field0_strange_feature_dif'] = df_test['field0_mod'] - df_test['field0'] 

# разница между месяцом покпуки и месяцом вылета
df_train['field2_3_dif'] = df_train[['field2', 'field3']].apply(lambda r: r['field3']-r['field2'] if r['field3']>=r['field2'] else 12-r['field2']+r['field3'], axis=1)
df_test['field2_3_dif'] = df_test[['field2', 'field3']].apply(lambda r: r['field3']-r['field2'] if r['field3']>=r['field2'] else 12-r['field2']+r['field3'], axis=1)

# признаки отношения и разницы field12 и field16
df_train['field_12_16_div'] = df_train['field12']/df_train['field16']
df_train['field_12_16_dif'] = df_train['field12'] - df_train['field16']
df_test['field_12_16_div'] = df_test['field12']/df_test['field16']
df_test['field_12_16_dif'] = df_test['field12'] - df_test['field16']

# признаки отношения field1 и field14
df_train['field_1_14_1_div'] = df_train['field1_1']/df_train['field14_1']
df_test['field_1_14_1_div'] = df_test['field1_1']/df_test['field14_1']

df_train['field_1_14_div'] = df_train['field1']/df_train['field14']
df_test['field_1_14_div'] = df_test['field1']/df_test['field14']

# среднее field12 и field16 по userid 
df_train[['field16_mean', 'field12_mean']] = df_train.groupby('userid')['field16', 'field12'].transform(np.mean)
df_test[['field16_mean', 'field12_mean']] = df_test.groupby('userid')['field16', 'field12'].transform(np.mean)

df_train['field16_mean_div'] = df_train['field16']/df_train['field16_mean']
df_train['field12_mean_div'] = df_train['field12']/df_train['field12_mean']
df_test['field16_mean_div'] = df_test['field16']/df_test['field16_mean']
df_test['field12_mean_div'] = df_test['field12']/df_test['field12_mean']

# mean encoding всех goal по field16
df_train['field16_cut'] = pd.cut(df_train['field16'], [-0.001,1,2,4,6,9,15,24,42,340])
df_test['field16_cut'] = pd.cut(df_test['field16'], [-0.001,1,2,4,6,9,15,24,42,340])
for targ in ['goal21', 'goal22', 'goal23', 'goal24', 'goal25', 'goal1']:
    map_dict = df_train.groupby('field16_cut')[targ].mean().to_dict()

    df_train['field16_mean_encoding_'+targ] = df_train['field16_cut'].map(map_dict).astype(np.float64)
    df_test['field16_mean_encoding_'+targ] = df_test['field16_cut'].map(map_dict).astype(np.float64)


df_train.drop('field16_cut', axis=1, inplace=True)
df_test.drop('field16_cut', axis=1, inplace=True)


# функция для составления признака - сочетание двух колонок
def all_mix_up(all_df, cols_to_mix):
    for i in tqdm(range(len(cols_to_mix))):
        for j in tqdm(range(i+1, len(cols_to_mix)), leave=False):
            c1 = cols_to_mix[i]
            c2 = cols_to_mix[j]
            new_col_name = c1 + '_mix_' + c2
            all_df[new_col_name] = all_df[c1].astype(str) + all_df[c2].astype(str)
            all_df[new_col_name] = all_df.groupby(new_col_name)['orderid'].transform('count')
    return all_df

# еще немного признаков
def get_encode_features(train_df, test_df, cols_to_mix):
    all_df = pd.concat([train_df, test_df], ignore_index=True)

    all_df['field_0_value_encode'] = all_df.groupby('field0')['orderid'].transform('count')

    all_df['field_12_value_encode'] = all_df.groupby('field12')['orderid'].transform('count')
    all_df['field_16_value_encode'] = all_df.groupby('field16')['orderid'].transform('count')

    all_df['field_12_value_encode_userid'] = all_df.groupby(['userid', 'field12'])['orderid'].transform('count')
    all_df['field_16_value_encode_userid'] = all_df.groupby(['userid', 'field16'])['orderid'].transform('count')

    all_df['field16_mix_12'] = (all_df['field12'].astype(str) + all_df['field16'].astype(str)).astype(int)
    all_df['field7_mix_17_25'] = (all_df['field7'].astype(str) + all_df['field17'].astype(str) + all_df['field25'].astype(str)).astype(int)
    all_df['field15_mix_24'] = (all_df['field15'].astype(str) + all_df['field24'].astype(str)).astype(int)
    all_df['field2_mix_3'] = (all_df['field2'].astype(str) + all_df['field3'].astype(str)).astype(int)

    all_df = all_mix_up(all_df, cols_to_mix)

    return all_df[all_df.userid.isin(train_df.userid.unique())], all_df[all_df.userid.isin(test_df.userid.unique())]

df_train, df_test = get_encode_features(df_train, df_test, fields_cols)

df_train.replace({np.inf:-999, -np.inf:-999}, inplace=True)
df_test.replace({np.inf:-999, -np.inf:-999}, inplace=True)

df_train.fillna(-999, inplace=True)
df_test.fillna(-999, inplace=True)


for col in df_train.columns:
    if df_train[col].dtype == np.float64:
        df_train[col] = np.round(df_train[col], 6)


# ### Вторая часть признаков, сделанная после НГ
def get_encode_features(train_df, test_df):
    all_df = pd.concat([train_df, test_df], ignore_index=True)

    # count encoding фичей по userid
    fields_for_counts = ['field1', 'field6', 'field12', 'field13', 'field16', 'field22']
    for col in fields_for_counts:
        all_df[col+'_value_counts'] = all_df.groupby(col)['userid'].transform('count')

    # mean encoding фичей по количеству билетов field15 и сумме field14
    fields_for_mean = ['field1_1', 'field22', 'field6', 'field13', 'field15', 'field17', 'field26', 
                       'indicator_goal21', 'indicator_goal22', 'indicator_goal23', 'indicator_goal24', 'indicator_goal25']
    for col in fields_for_mean:
        all_df[col+'_mean_tickets'] = all_df.groupby(['field15', 'field14'])[col].transform('mean')
        all_df[col+'_mean_tickets_div'] = all_df[col+'_mean_tickets']/all_df[col]

    
    # процент билетов для взрослых
    all_df['tickets_grown_perc'] = all_df['field24']/all_df['field15']
    # цена одного билета
    all_df['price_of_tickects'] = all_df['field1_1']/all_df['field15']
    # процент билетов для детей
    all_df['tickets_child_perc'] = (all_df['field28'])/all_df['field15']

    # функции добавляют mean encoding переменных по другим переменным
    all_df = get_add_encode_mean(all_df)
    all_df = get_encode_prices_mean(all_df)

    return all_df[all_df.userid.isin(train_df.userid.unique())], all_df[all_df.userid.isin(test_df.userid.unique())]


def get_add_encode_mean(all_df):
    features_to_encode = ['field1', 'field14', 'field25', 'field26', 'field27']
    features_by_encode = ['field1', 'field14', 'field25', 'field26', 'field27', 'field4']
    for feat_by in tqdm(features_by_encode):
        for feat_to in features_to_encode:
            if feat_to != feat_by:
                all_df[feat_to+'_mean_encode_' + feat_by] = all_df.groupby(feat_by)[feat_to].transform('mean')
                all_df[feat_to+'_mean_encode_' + feat_by+'_div'] = all_df[feat_to]/all_df[feat_to+'_mean_encode_' + feat_by]
                gc.collect()
    return all_df


def get_encode_prices_mean(all_df):
    # делаем энкодинг по всем фичам + userid
    features_by_encode = copy.copy(fields_cols) + ind_cols + ['userid']
    features_by_encode.remove('field1_1')
    features_by_encode.remove('field14_1')
    # энкодинг для 5 самых важных фич
    features_to_encode = ['field1', 'field14', 'field27', 'field16', 'field12']
    for feat_by in tqdm(features_by_encode):
        for feat_to in features_to_encode:
            new_col = feat_to+'_mean_encode_' + feat_by
            # если такая колонка уже есть (сделана с помощью get_add_encode_mean) не повторяем вычисления
            if feat_to != feat_by and new_col not in all_df.columns :
                all_df[new_col] = all_df.groupby(feat_by)[feat_to].transform('mean')
                all_df[new_col+'_div'] = all_df[feat_to]/all_df[new_col]
                gc.collect()
    return all_df


# функция для создания второй части признаков
df_train, df_test = get_encode_features(df_train, df_test)

gc.collect()


# ### Предсказания модели
# загружаем названия отобранных фич
with open(os.path.join(data_folder, 'feature_for_task_1.txt'), 'r') as f:
    features = json.load(f)

variable_names = features['variable_names']
len(variable_names)

for col in tqdm(df_train.columns):
    df_train[col] = df_train[col].replace([np.inf, -np.inf], -9999)
    df_train[col] = df_train[col].fillna(-999)

    df_test[col] = df_test[col].replace([np.inf, -np.inf], -9999)
    df_test[col] = df_test[col].fillna(-999)

params = {'colsample_bytree': 0.8856,
 'learning_rate': 0.0113,
 'max_depth': 5,
 'n_estimators': 400,
 'num_leaves': 59,
 'reg_lambda': 0.1124,
 'subsample': 0.599}
 
lgb_cl = lgb.LGBMClassifier(**params, random_state=10, n_jobs=-1)

gc.collect()

lgb_cl.fit(df_train[variable_names], df_train['goal1'])

scr = lgb_cl.predict_proba(df_test[variable_names])[:,1]

df_test['proba'] = scr

df_test[['orderid', 'proba']].to_csv(os.path.join(data_folder, 'alwayswannadie_1.csv'), index=False)

