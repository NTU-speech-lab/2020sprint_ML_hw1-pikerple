import numpy as np
import pandas as pd
import sys
np.set_printoptions(threshold=np.inf)
w = np.load('w_best.npy')
w = np.load('b_best.npy')
def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X 
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)
with open(sys.argv[5]) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
    X_test = np.pad(X_test,((0,0),(0,8)),'constant')
for i in range(X_test.shape[0]):
    age = X_test[i][0]
    data = X_test[i][-3]
    if(0<=age<=15):
        X_test[i][-5]=1
    elif(16<=age<=30):
        X_test[i][-4]=1
    elif(31<=age<=50):
        X_test[i][-3]=1
    elif(51<=age<=70):
        X_test[i][-2]=1
    else:
        X_test[i][-1]=1
    
    if(0<=data<=15):
        X_test[i][-6]=1
    elif(16<=data<=45):
        X_test[i][-7]=1
    else:
        X_test[i][-8]=1
# Predict testing labels
predictions = _predict(X_test, w, b)
with open(sys.argv[6], 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))