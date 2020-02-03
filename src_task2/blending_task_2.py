import pandas as pd
import numpy as np

alwayswannadie = pd.read_csv('submissions/alwayswannadie_2.csv')
datasanta = pd.read_csv('submissions/datasanta_2.csv')
egor = pd.read_csv('submissions/egor_2.csv')

targets = ['goal21', 'goal22', 'goal23', 'goal24', 'goal25']
weights = [(0.6, 0.2, 0.2), (0.6, 0.3, 0.1), (0.6, 0.2, 0.2), (0.6, 0.2, 0.2), (0.6, 0.2, 0.2)]

final = datasanta.merge(alwayswannadie, on=['orderid'], suffixes=['_ds', '_al'])

for targ in targets:
    egor.rename({targ: targ+'_eg'}, axis=1, inplace=True)

final = final.merge(egor, on=['orderid'])

for i in range(len(targets)):
    
    targ = targets[i]
    weight = weights[i]
    
    final[targ] = np.round((weight[0]*final[targ+'_al'] + weight[1]*final[targ+'_ds'] + weight[2]*final[targ+'_eg']),4)

final[['orderid', 'goal21', 'goal22', 'goal23', 'goal24', 'goal25']].to_csv('submissions/final_submission_task_2.csv', index=False)

