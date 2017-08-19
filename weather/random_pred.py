from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train_data = pd.read_csv('churn_train.csv')
label = train_data['label']
check = pd.read_csv('check.csv')

check['phnum'] = check['phnum'].apply(lambda x: x.strip())

randomForest = RandomForestClassifier(n_estimators=1000)
randomForest.fit(train, label)
dtest = randomForest.predict(test)
result = pd.read_csv('check.csv')
output = pd.DataFrame(data={"label": dtest, "phnum": check['phnum']})
output.to_csv('result.csv')
