import pandas as pd
import numpy as np
import lightgbm as lgb
import itertools


def ratio(data):
    """
    признаки отношения
    
    значение поля по текущей заявке делится
    на среднее значение поля для этого юзера (среднее вычисляется по всем заявкам юзера)
    
    """
    groupvar = 'userid'
    
    datavars = ['field0',
                'field1',
                'field12', 
                'field13',
                'field14',
                'field16',
                'field22',
                'field26',
                'field16_minus_field0_next']
    
    fnames = []
    
    for datavar in datavars:

        fname = '{}_div_{}_mean'.format(datavar, datavar)

        data[fname] = data[datavar] / data.groupby([groupvar])[datavar].transform(np.mean) 

        fnames += [fname]

    data = data.replace({np.inf: np.nan, -np.inf: np.nan})
        
    return data, fnames


def histogram(data, train):
    """
    value_counts кодирование

    Значение переменной заменяется частотой встречаемости в тренировочном наборе
    Частота считается только по данным train. При расчете признаков для test, значения,     
    которые не встретились в train, но есть в test заменяются на 1.
    """
    
    datavars = ['field0', 
                'field12', 
                'field16',
                'field6', 
                'field13',
                'field22',
               ]
    
    fnames = []

    for fname in datavars:
    
        frequency_table = pd.DataFrame(train[fname].value_counts()).to_dict()[fname]
        
#       При расчете признаков для test, значения,     
#       которые не встретились в train, но есть в test заменяются на 1.
        frequency_table_for_unknown = {k: 1 for k in list(set(data[fname]).difference(set(frequency_table.keys())))}

        frequency_table.update(frequency_table_for_unknown)

        data[fname + '_value_counts'] = data[fname].map(frequency_table)
        
        fnames += [fname + '_value_counts']

    data['goal12345_count'] = data['indicator_goal21'].astype(str) +     data['indicator_goal22'].astype(str) +     data['indicator_goal23'].astype(str) +     data['indicator_goal24'].astype(str) +     data['indicator_goal25'].astype(str)
    freqs = pd.DataFrame(data['goal12345_count'].value_counts()).to_dict()['goal12345_count']
    data['goal12345_count'] = data['goal12345_count'].map(freqs)

    return data, fnames + ['goal12345_count']


def interactions2(data):
    """
    признаки состояний для пар индикаторов
    
    45_10 - если 4 индикатор установлен в 1, 
                 5 индикатор установлен в 0, 
            то 45_10=1, иначе 45_10=0
    
    35_00 - если 3 индикатор установлен в 0, 
                 5 индикатор установлен в 0, 
            то 35_00=1, иначе 35_00=0
    """
    
    t = [1, 2, 3, 4, 5]
    
    goals12 = list(itertools.combinations(t, 2))
    
    fnames = []
    
    for goal1, goal2 in goals12:
        
        fname = '{}{}'.format(goal1, goal2)
        
        fnames += [fname]
        
        data[fname] = data['indicator_goal2' + str(goal1)].astype(str) +         data['indicator_goal2' + str(goal2)].astype(str)

    dummies = pd.get_dummies(data[fnames])
    
    data = pd.concat([data, dummies], axis=1)
    
    fnames = ['45_10', '35_10', '35_00', '24_11', '34_11', '24_00',
             '14_10', '14_11', '34_01', '34_00', '35_11', '35_01',
             '23_10', '13_10', '25_11', '45_01', '45_00', '45_11',
             '15_11', '24_10', '25_10', '13_11', '23_00', '34_10',
             '12_00',]
    
    return data, fnames
	

def shift(data):
    """
    разница между значением поля текущей и следующей заявки юзера
    """
    data = data.sort_values(by='field4')
        
    shiftcols = ['field' + str(i) for i in range(30) if i not in (4,29)]
    
    fnames = []
    
    for col in shiftcols:
        
        fname = col + '_diff_1'
        
        data[fname] = data[col] - data.groupby(['userid'])[col].shift(-1)
        
        fnames += [fname]  
        
    data.fillna(-999, inplace=True)

    return data, fnames


def raw(data):
    """
    сырые признаки без каких либо преобразований
    """
    features = ['field' + str(i) for i in range(30) if i not in (9,29,)] + [
        'indicator_goal23',
        'indicator_goal24',
        'indicator_goal25'
    ]

    return data, features
	

def group(data, aggregations, groupvar='userid'):
   
    data_agg = data.groupby(groupvar).agg(aggregations)

    data_agg.columns = pd.Index(['{}_{}_{}'.format(e[0], groupvar, e[1])
                               for e in data_agg.columns.tolist()])

    data_agg = data_agg.reset_index()

    data = data.merge(data_agg, how='left', on=groupvar)
    
    fnames = [col for col in data_agg.columns.tolist() if col != 'userid']
    
    return data, fnames

	
def statistics(data):
    """
    заявки аггрегируются по юзеру, 
    после аггрегирования вычисляется среднее (mean) и сумма (sum) 
    для полей indicator_goal23, indicator_goal24, indicator_goal25
    """
    aggregations = {'indicator_goal23': ['mean', 'sum',],
                    'indicator_goal24': ['mean', 'sum',],
                    'indicator_goal25': ['mean', 'sum',]}
    
    data, fnames = group(data, aggregations, groupvar='userid')

    return data, fnames


def preprocessing(data):
    
    data.loc[data.field21==1, 'field3'] = data['field3'][data.field21==1] + 12
    
    return data


def magic(data):
    """
    признаки на основе полей 0 и 16
    """
    data = data.sort_values(by='field4')

    fnames = []
    
    f1, f2 = ('field0', 'field16')

    # разница в днях между датой вылета по текущей заявке 
    # и датой следующей заявки
    data['{}_minus_{}_next'.format(f2, f1)] = data[f2] -         data.groupby(['userid'])[f1].shift(-1)
    fnames += ['{}_minus_{}_next'.format(f2, f1)]
    
    #
    data['{}_div_{}_minus_{}_next'.format(f2, f2, f1)] =         data.groupby(['userid'])[f2].shift(-1)  / (data[f2] -         data.groupby(['userid'])[f1].shift(-1))        
    fnames += ['{}_div_{}_minus_{}_next'.format(f2, f2, f1)]

    #
    data['{}_minus_{}_prev'.format(f2, f1)] = data[f2] -         data.groupby(['userid'])[f1].shift(1)
    fnames += ['{}_minus_{}_prev'.format(f2, f1)]

    #
    data['{}_minus_{}'.format(f2, f1)] = data[f2] - data[f1]
    fnames += ['{}_minus_{}'.format(f2, f1)]

    #
    data['{}_minus_{}_past_plus_{}'.format(f2, f2, f1)] = data[f2] -     (data.groupby(['userid'])[f2].shift(-1) + data[f1])
    fnames += ['{}_minus_{}_past_plus_{}'.format(f2, f2, f1)]
    
    return data, fnames


train = pd.read_csv('data\onetwotrip_challenge_train.csv')
test = pd.read_csv('data\onetwotrip_challenge_test.csv')

timevar = 'orderid'
target = 'goal22'

train = preprocessing(train)
test = preprocessing(test)

train, raw_f = raw(train)
test, raw_f = raw(test)

train, shift_f = shift(train)
test, shift_f = shift(test)

train, magic_f = magic(train)
test, magic_f = magic(test)

train, value_counts_f = histogram(train, train)
test, value_counts_f = histogram(test, train)

train, statistics_f = statistics(train)
test, statistics_f = statistics(test)

train, ratio_f = ratio(train)
test, ratio_f = ratio(test)

train, interactions_f2 = interactions2(train)
test, interactions_f2 = interactions2(test)

bad_users = ('0278bd647e2d9db5a5c342c3b2d8ff8ef484e181f51a726adb4077842cb35792',
                '0846b1fb28f8de9779f4a3fda6dedee5dfee657cffdccfaed89ec0a5128bbe11',
                '97826bf9f43bd3543a9b615df67bd19979847d8a8401ca54b652a3d9be632965',
                '98e3b9eba8259c0fe0599cc0490a39e8d946942abcd43fa28ba1d2872f2a462b')
    
train = train[~train.userid.isin(bad_users)]

features = [
            *raw_f,
            *shift_f,
            *magic_f,
            *value_counts_f,
            *statistics_f,
            *ratio_f,
            *interactions_f2,
]


features = [f for f in features if f not in ('24_00',
                                             'field28_diff_1',
                                             'indicator_goal24',
                                             '14_10',
                                             'indicator_goal23',
                                             '14_11',
                                             'indicator_goal25',
                                             '24_10',
                                             'field21_diff_1',
                                             '25_10',
                                             '34_10',
                                             'field9_diff_1',
                                             '12_00')]

train = train.sort_values(timevar).reset_index(drop=True)


best_iter = 600

sub = test

goal = 'goal22'
    
params = {'learning_rate': 0.038, 
          'min_data_in_leaf': 1000, 
          'reg_lambda': 0.01, 
          'reg_alpha': 1.4,
          "metric": 'auc',
          "verbosity": -1,
          'random_state': 47,
          'n_estimators': 600
 }
    
clf = lgb.LGBMClassifier(**params, num_boost_round=best_iter)

clf.fit(train[features], train[goal])

Y_pred = clf.predict_proba(test[features])[:, 1]

sub.loc[:, goal] = np.round(Y_pred, 4)

sub27 = pd.read_csv(r'submissions\datasanta_2.csv')

sub = sub.merge(sub27, on='orderid')

sub.loc[:, 'goal22'] = sub['goal22_x']

sub[['orderid', 'goal21', 'goal22', 
     'goal23',
     'goal24', 'goal25']].to_csv(r'submissions\datasanta_2.csv', index=False)

