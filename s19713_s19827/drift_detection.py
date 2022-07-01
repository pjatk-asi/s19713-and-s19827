import numpy as np
import pandas as pd
from datetime import date, datetime
from pickle import TRUE
import os.path
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.simplefilter(action='ignore',category=UserWarning)

#Read model evaluation results
evaluation_results = pd.read_csv('evaluation/model_eval.csv', parse_dates=['time_stamp'], dayfirst=True)
last_run = evaluation_results['time_stamp'].max()
print(last_run)

#Prepare data for tests
ACC_logs = evaluation_results[evaluation_results['metric'] == 'ACC']
last_ACC = ACC_logs[ACC_logs['time_stamp']==last_run]['score'].values[0]
all_other_ACC = ACC_logs[ACC_logs['time_stamp']!=last_run]['score'].values

#Hard test
hard_test_ACC = last_ACC > np.mean(all_other_ACC)
print('Is data drift detected?')
print('ACC: ', hard_test_ACC)

#For ACC, we identify drift (print TRUE) if the new ACC is smaller than the mean of all the past ACC
hard_test_ACC = last_ACC < np.mean(all_other_ACC)

#Parametric test
param_test_ACC = last_ACC < np.mean(all_other_ACC) - 2*np.std(all_other_ACC) #Średnia poprzednich pomiarów powiększona o dwukrotność standardowego odchylenia

print('\n.. Parametric test ..')
print('ACC: ', param_test_ACC)

#Non-parametric (IQR) test
#For ACC, we identify drift (print TRUE) if the new ACC is larger than te 3rd quantile + 1.5 IQR
iqr_ACC = np.quantile(all_other_ACC, 0.75) - np.quantile(all_other_ACC, 0.25)
iqr_test_ACC = last_ACC < np.quantile(all_other_ACC, 0.25) - iqr_ACC*1.5

print('\n.. IQR test ..')
print('ACC: ', iqr_test_ACC, '  R2: ', iqr_test_ACC)

#Re-training signal
drift_df = pd.DataFrame()
drift_signal_file = 'evaluation/model_drift.csv'
now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

print('\n  --- DRIFT DETECTION ---')

actual_tests = {
                            'hard_test_ACC': hard_test_ACC, 
                            'param_test_RMSE': param_test_ACC, 
                            'iqr_test_RMSE': iqr_test_ACC,    
               }


a_set = set(actual_tests.values())
if False in set(actual_tests.values()):
    drift_detected = TRUE

if drift_detected:
    print('There is a DRIFT detected in...')
    for a in actual_tests:
        if actual_tests[a]:
            print(a)
    drift_df = drift_df.append({'time_stamp': now, 'model_name': evaluation_results, 
                            'hard_test_ACC': str(hard_test_ACC),
                            'param_test_ACC': str(param_test_ACC),
                            'iqr_test_ACC': str(iqr_test_ACC),
                            }, ignore_index=True)



 # Save drift signal to file    
    if os.path.isfile(drift_signal_file):
        drift_df.to_csv(drift_signal_file, mode='a', header=False, index=False)
    else:
        drift_df.to_csv(drift_signal_file, index=False)
else:
    print('There is NO DRIFT detected.')

if drift_detected:
    import subprocess
    print('\n  --- RE-TRAINING ---\n')
    subprocess.call(['python', 'train.py'])