import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

train_data = pd.read_csv('churn_train.csv')
test_data = pd.read_csv('check.csv')
print(test_data.head())

def substrings_in_string(subString):
    c = subString[1:4]
    return c

def change_String(substrings):
	if (substrings == 'True'):
		return 1
	else:
		return 0

train_data['phone'] = train_data['arcode'].map(lambda x: substrings_in_string(x))
test_data['phone'] = test_data['phnum'].map(lambda x: substrings_in_string(x))
# print(train_data.groupby(['phone'])['phnum'].count())

Intplan = pd.get_dummies(train_data.intplan)
Phnum = pd.get_dummies(train_data.phone)
train_data_ver = pd.concat([Phnum, Intplan], axis=1)
train_data_ver['Voice'] = train_data['voice']
train_data_ver['Tdimn'] = train_data['tdcal']
train_data_ver['Tdchar'] = train_data['tdchar']
train_data_ver['Temin'] = train_data['temin']
train_data_ver['Tnmin'] = train_data['tnmin']
train_data_ver['Tncal'] = train_data['tncal']
train_data_ver['Tnchar'] = train_data['tnchar']
train_data_ver['Tical'] = train_data['tical']
cols = ['Voice', 'Tdimn', 'Tdchar', 'Temin', 'Tnmin', 'Tncal', 'Tnchar', 'Tical']
train_data_ver[cols] = train_data_ver[cols].apply(lambda x: (x-np.mean(x)) / (np.max(x) - np.min(x)))

Intplan = pd.get_dummies(test_data.intplan)
Phnum = pd.get_dummies(test_data.phone)
test_data_ver = pd.concat([Phnum, Intplan], axis=1)
test_data_ver['Voice'] = test_data['voice']
test_data_ver['Tdimn'] = test_data['tdcal']
test_data_ver['Tdchar'] = test_data['tdchar']
test_data_ver['Temin'] = test_data['temin']
test_data_ver['Tnmin'] = test_data['tnmin']
test_data_ver['Tncal'] = test_data['tncal']
test_data_ver['Tnchar'] = test_data['tnchar']
test_data_ver['Tical'] = test_data['tical']
test_data_ver[cols] = test_data_ver[cols].apply(lambda x: (x-np.mean(x)) / (np.max(x) - np.min(x)))

label = train_data['label']


# randomForest = RandomForestClassifier(n_estimators=1000)
# randomForest.fit(train_data_ver, label)
# print("10 cross_val_score=", np.mean(cross_val_score(randomForest, train_data_ver, label, cv=10)))
# dtest = randomForest.predict(test_data_ver)
# dtest.to_csv('result.csv')
