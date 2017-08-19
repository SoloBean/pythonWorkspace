import numpy as np
import xgboost as xgb

modelfile = 'weather.model'
outfile = 'weather_pred.csv'

dtest = np.loadtxt('test_data.csv', delimiter=',', skiprows=1)
data = dtest[:, 1:106]

print('finish loading from csv')
xgmat = xgb.DMatrix(data)
bst = xgb.Booster(model_file='weather.model')
ypred = bst.predict(xgmat)

