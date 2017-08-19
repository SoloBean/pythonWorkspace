import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def convert_float(str):
    if(str == 'True'):
        return 1
    else:
        return 0

dtrain = np.loadtxt('train_data.csv', delimiter=',', skiprows=1, converters={106: lambda x: convert_float(x)})
print('finish loading from csv')

label = dtrain[:, 106]
data = dtrain[:, 0:105]

xgmat = xgb.DMatrix( data, label=label)
params = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'subsample': 0.8,
    'colsample_bytree': 0.85,
    'eta': 0.05,
    'max_depth': 7,
    'seed': 2016,
    'silent': 0,
    'eval_metric': 'rmse'
}
watchlist = [(xgmat, 'train')]
num_round = 500
bst = xgb.train(params, xgmat, num_round, watchlist)
bst.save_model('weather.model')
print('finish training')
