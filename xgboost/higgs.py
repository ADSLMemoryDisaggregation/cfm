import xgboost as xgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import threading
import time

def train_and_test(filename, param, num_round, nthread):
    #load data
    start_time = time.time()
    data = pd.read_csv(filename, header=None)
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    X = X.astype('float64')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param['nthread'] = nthread
    bst = xgb.train(param, dtrain, num_round)
    preds = bst.predict(dtest)
    end_time = time.time()
    print(filename + ': ' + 'AUC:{} Time:{}'.format(roc_auc_score(y_test, preds), end_time - start_time))


param = {'max_depth': 9, 'eta': 0.03, 'objective': 'binary:logistic'}

# Start two threads to train the models
thread1 = threading.Thread(target=train_and_test, args=('/mydata/HIGGS.csv', param, 245, 1))
thread2 = threading.Thread(target=train_and_test, args=('/mydata/SUSY.csv', param, 500, 1))

thread1.start()
thread2.start()

thread1.join()
thread2.join()
