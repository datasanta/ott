import pandas as pd
import numpy as np

datasanta = pd.read_csv('submissions/datasanta_1.csv')
egor = pd.read_csv('submissions/egor_1.csv')
alwayswannadie = pd.read_csv('submissions/alwayswannadie_1.csv')

datasanta.rename({'proba':'proba_ds'}, axis=1, inplace=True)
egor.rename({'proba':'proba_eg'}, axis=1, inplace=True)
alwayswannadie.rename({'proba':'proba_al'}, axis=1, inplace=True)

final = datasanta.merge(egor)
final = final.merge(alwayswannadie)

final['proba'] = np.round(0.5*final['proba_ds'] + 0.3*final['proba_eg'] + 0.2*final['proba_al'],4)

final[['orderid', 'proba']].to_csv('submissions/final_submission_task_1.csv', index=False)

