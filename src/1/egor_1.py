import pandas as pd
import numpy as np
import lightgbm as lgb
from itertools import combinations
from sklearn.base import clone

#Загрузка данных
df = pd.read_csv(r'./data/onetwotrip_challenge_train.csv')
test = pd.read_csv(r'./data/onetwotrip_challenge_test.csv')

def feature_engineering(data):
    data = data.sort_values(by='field4')
    #Кол-во записей по пользователю
    data['user_count'] = data.groupby('userid')['userid'].transform('count')
    #Исправленные месяца
    data['month_p'] = np.where(data['field21']==1, data['field2']+12, data['field2'])
    data['month_f'] = np.where((data['field2'] == data['field3']) & (data['field16']<=31), data['month_p'],
                    np.where(data['field3']<data['field2'], data['field3']+12,
                    np.where(data['field16']<365, data['field3'], data['field3']+12)))   
    data['same_flight_count_month'] = data.groupby(['userid', 'month_f'])['userid'].transform('count')
    #Время до полета
    data['hours_to_flight'] = (data['field16']-1)*24+(24-data['field11'])+data['field23']
    #Расстояние до ближайших заказов пользователя
    data['hours_since_purchase'] = (data['field0']-1)*24+(24-data.groupby('userid')['field11'].shift(1))+data['field11']
    data['hours_to_next_purchase'] = data.groupby('userid')['hours_since_purchase'].shift(-1)
    ##Расстояние между этой датой полета и датой полета соседних покупок
    data['diff_hours_between_next_purchase'] = data['hours_to_flight']     - (data['hours_to_next_purchase'] + data.groupby('userid')['hours_to_flight'].shift(-1))
    data['diff_hours_between_prev_purchase'] = data.groupby('userid')['hours_to_flight'].shift(1)     - (data['hours_since_purchase'] + data['hours_to_flight'])

    #Дополнительные шифты по колонкам
    cols_shift1 = ['field13', 'field16', 'field22', 'field26']
    for i in cols_shift1:
        data['sh_{}_p_d'.format(i)] = data[i] - data.groupby('userid')[i].shift(1)
        data['sh_{}_n'.format(i)] = data.groupby('userid')[i].shift(-1)
        data['sh_{}_n_d'.format(i)] = data[i] - data['sh_{}_n'.format(i)]
        cols_shift2 = ['field16', 'field26']
    for i in cols_shift2:
        data['sh_{}_n1'.format(i)] = data.groupby('userid')[i].shift(-2)
        data['sh_{}_n_d1'.format(i)] = data[i] - data['sh_{}_n1'.format(i)]

    #Допонительные фичи по пользователям
    cols_to_user_features = ['field2', 'field3', 'field16', 'field26']
    for i in cols_to_user_features:
        data['{}_um'.format(i)] = data[i] - data.groupby('userid')[i].transform('mean')
        data['{}_us'.format(i)] = data.groupby('userid')[i].transform('std') 

    #Дополнительные фичи по пользователям
    for i in ['indicator_goal24']:
        data['{}_std'.format(i)] = data.groupby('userid')[i].transform('std')
        
    return data

def target_encoding(data, target, column, z=10, l=100):
    mean_all = data[target].mean()
    mean_count = data.groupby(column)[target].agg(['mean', 'count'])
    mean_count = mean_count[mean_count['count']>z]
    mean_count['mean'] = (mean_count['mean']*mean_count['count'] + l*mean_all) / (l+mean_count['count'])
    return mean_count['mean'].to_dict() 

def target_encoding_fit_transform(train, test, target, column, new_column, z=10, l=100):
    tedict = target_encoding(train, target, column, z, l)
    temean = train[target].mean()
    test[new_column] = test[column].map(tedict).fillna(temean)
    train[new_column] = train[column].map(tedict).fillna(temean) 
    return train, test

df = feature_engineering(df)
test = feature_engineering(test)
df, test = target_encoding_fit_transform(df, test, 'goal1', 'field12', 'field12_e', z=1000, l=1000)

cols_forward = ['hours_to_flight', 'diff_hours_between_next_purchase', 'field12', 'field14', 'sh_field22_n_d', 'field6', 'hours_to_next_purchase', 
                'month_p', 'sh_field26_n', 'field20', 'indicator_goal22', 'sh_field26_p_d', 'sh_field16_n_d1', 'indicator_goal24_std']

cols_backward = ['month_p', 'sh_field13_n_d', 'sh_field16_p_d', 'field1', 'hours_to_next_purchase', 'field6', 'sh_field22_n_d',
                 'field16', 'field12', 'diff_hours_between_next_purchase', 'field21', 'field28', 'indicator_goal24', 'sh_field26_n1',
                 'field2_us', 'field3_us', 'field16_um', 'field26_us']

cols = ['field0', 'field1', 'field2', 'field3', 'field4', 'field5', 'field6', 'field7', 'field8', 'field9', 'field10', 'field11', 'field12',
        'field13', 'field14', 'field15', 'field16', 'field17', 'field18', 'field19', 'field20', 'field21', 'field22', 'field23', 'field24', 'field25',
        'field26', 'field27', 'field28', 'field29', 'indicator_goal21', 'indicator_goal22', 'indicator_goal23', 'indicator_goal24', 'indicator_goal25', 'field12_e']


model1 = lgb.LGBMRegressor(objective='binary',
                          learning_rate=0.01238, 
                          n_estimators=950,
                          feature_fraction=0.6094,
                          num_leaves=15, 
                          max_depth=6,
                          n_jobs=-1, 
                          min_data_in_leaf=50)

model2 = lgb.LGBMRegressor(objective='binary',
                          learning_rate=0.068, 
                          max_depth=12, 
                          n_estimators=200, 
                          num_leaves=10,
                          boosting_type='goss')

#Обучение моделей
fm1 = clone(model2).fit(df.loc[:,cols_forward], df['goal1'])
fm2 = clone(model1).fit(df.loc[:,cols_forward], df['goal1'])
bm1 = clone(model2).fit(df.loc[:,cols_backward], df['goal1'])
bm2 = clone(model1).fit(df.loc[:,cols_backward], df['goal1'])
m1 = clone(model1).fit(df.loc[:,cols], df['goal1'])

#Предикт
test['proba'] = fm1.predict(test.loc[:,cols_forward])*0.5*0.5 + fm2.predict(test.loc[:,cols_forward])*0.5*0.5 + bm1.predict(test.loc[:,cols_backward])*0.3*0.5 + bm2.predict(test.loc[:,cols_backward])*0.3*0.5 + m1.predict(test.loc[:,cols])*0.2

test[['orderid', 'proba']].to_csv(r'.\submissions\egor_1.csv', index=False)