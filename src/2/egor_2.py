
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from itertools import combinations
from sklearn.base import clone


# In[2]:


# Загрузка данных
df = pd.read_csv(r'data/onetwotrip_challenge_train.csv')
test = pd.read_csv(r'data/onetwotrip_challenge_test.csv')


# In[3]:


cols = ['field0', 'field1', 'field2', 'field3', 'field4', 'field5', 'field6', 'field7', 'field8', 'field9',
       'field10', 'field11', 'field12', 'field13', 'field14', 'field15', 'field16', 'field17', 'field18',
       'field19', 'field20', 'field21', 'field22', 'field23', 'field24', 'field25', 'field26', 'field27',
       'field28', 'field29']

indicators = ['indicator_goal21', 'indicator_goal22', 'indicator_goal23', 'indicator_goal24', 'indicator_goal25']

goals = ['goal21', 'goal22', 'goal23', 'goal24', 'goal25']

cols_model1 = cols + indicators + ['user_count', 'month_p', 'month_f', 'same_flight_count_month', 'hours_to_flight', 'hours_since_purchase', 'hours_to_next_purchase',
 'diff_hours_between_next_purchase', 'diff_hours_between_prev_purchase', 'sh_field14_p_d', 'sh_field14_n_d', 'sh_field27_p_d',
 'sh_field27_n_d', 'sh_field22_p_d', 'sh_field22_n_d', 'sh_field26_p_d', 'sh_field26_n_d', 'sh_field1_p_d', 'sh_field1_n_d',
 'sh_field16_p_d', 'sh_field16_n_d', 'sh_field12_p_d', 'sh_field12_n_d', 'sh_field13_p_d', 'sh_field13_n_d', 'sh_field0_p_d',
 'sh_field0_n_d', 'field14_um', 'field27_um', 'field22_um', 'field26_um', 'field1_um', 'field16_um', 'field12_um', 'field13_um',
 'field0_um', 'indicator_goal21_sum', 'indicator_goal22_sum', 'indicator_goal23_sum', 'indicator_goal24_sum', 'indicator_goal25_sum']

cols_model2 = cols + ['indicator_goal21indicator_goal22', 'indicator_goal21indicator_goal23', 'indicator_goal21indicator_goal24',
'indicator_goal21indicator_goal25', 'indicator_goal22indicator_goal23', 'indicator_goal22indicator_goal24',
'indicator_goal22indicator_goal25', 'indicator_goal23indicator_goal24', 'indicator_goal23indicator_goal25',
'indicator_goal24indicator_goal25'] + \
indicators 

model = lgb.LGBMRegressor(objective='binary',
                          learning_rate=0.01238, 
                          n_estimators=1200,
                          feature_fraction=0.6094,
                          subsample=0.6,
                          num_leaves=15, 
                          max_depth=6,
                          random_state=10,
                          n_jobs=-1, 
                          min_data_in_leaf=50)


# In[4]:


def target_encoding_get_dict(data, target, column, z=10, l=100):
    """Подготовка словаря для таргет энкодинга"""
    mean_all = data[target].mean()
    mean_count = data.groupby(column)[target].agg(['mean', 'count'])
    mean_count = mean_count[mean_count['count']>z]
    mean_count['mean'] = (mean_count['mean']*mean_count['count'] + l*mean_all) / (l+mean_count['count'])
    return mean_count['mean'].to_dict() 

def target_encoding_fit_transform(train, test, target, column, new_column, z=10, l=100):
    """Обучение и предискиты для словаря таргет энкодинга"""
    tedict = target_encoding_get_dict(train, target, column, z, l)
    temean = train[target].mean()
    test[new_column] = test[column].map(tedict).fillna(temean)
    train[new_column] = train[column].map(tedict).fillna(temean) 
    return train, test

def target_encoding(train, test, goals):
    """Применение таргет энкодинга к нужным колонкам"""
    cols_to_encoding = ['field12', 'field27']
    for i in cols_to_encoding:
        for j in goals:
            train, test = target_encoding_fit_transform(train, test, j, i, '{}_e_{}'.format(i, j), z=1000, l=1000)
    return train, test

def feature_engineering(data):
    """Генерируем основные фичи"""
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
    #Расстояние между этой датой полета и датой полета соседних покупок
    data['diff_hours_between_next_purchase'] = data['hours_to_flight']     - (data['hours_to_next_purchase'] + data.groupby('userid')['hours_to_flight'].shift(-1))
    data['diff_hours_between_prev_purchase'] = data.groupby('userid')['hours_to_flight'].shift(1)     - (data['hours_since_purchase'] + data['hours_to_flight'])

    #Разница между текущем знчаемем признака и соседними значениями для пользователя
    add_cols = ['field14', 'field27', 'field22', 'field26', 'field1', 'field16', 'field12', 'field13', 'field0']
    for i in add_cols:
        data['sh_{}_p_d'.format(i)] = data[i] - data.groupby('userid')[i].shift(1)
        data['sh_{}_n_d'.format(i)] = data[i] - data.groupby('userid')[i].shift(-1)

    #Разница между текущим значениме признака и средним значением для пользователя
    for i in add_cols:
        data['{}_um'.format(i)] = data[i] - data.groupby('userid')[i].transform('mean')

    #Суммы индикаторов по пользователю
    iteractions = ['indicator_goal21', 'indicator_goal22', 'indicator_goal23', 'indicator_goal24', 'indicator_goal25']
    for i in iteractions:
        data['{}_sum'.format(i)] = data.groupby('userid')[i].transform('sum')

    #Итеракции по индикаторам
    for i, j  in [*combinations(iteractions, 2)]:
        data['{}{}'.format(i, j)] = data.loc[:,[j, i]].min(axis=1)

    return data

def model_fit_predict(train, test, model, cols_model1, cols_model2, goals):
    """Обучение моделей и предикиты
    Предикт для каждого таргета - это блендинг двух моделей.
    Первая - модель без шифтов, но с таргет энкодингом для двух фич.
    Вторая - модель без таргет энкодинга но с шифтами и др. сатистисками по пользователям """
    for i in goals:
        model1 = clone(model).fit(train.loc[:,cols_model1], train[i])
        model2 = clone(model).fit(train.loc[:,cols_model2+['field12_e_{}'.format(i), 'field27_e_{}'.format(i)]], train[i])
        model1_p = model1.predict(test.loc[:,cols_model1]) 
        model2_p = model2.predict(test.loc[:,cols_model2+['field12_e_{}'.format(i), 'field27_e_{}'.format(i)]])
        test[i] = np.where(test['user_count']==1, model1_p*0.2 + model2_p*0.8, 
                  np.where(test['user_count']<3, model1_p*0.5 + model2_p*0.5, model1_p*0.7 + model2_p*0.3))
    return test


# In[5]:


df = feature_engineering(df)
test = feature_engineering(test)
df, test = target_encoding(df, test, goals)


# In[6]:


test = model_fit_predict(df, test, model, cols_model1, cols_model2, goals)


# In[7]:


test[['orderid'] + goals].to_csv(r'submissions\egor_2.csv', index=False)

