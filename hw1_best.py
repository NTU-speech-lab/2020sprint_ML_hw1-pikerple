import sys
import pandas as pd
import numpy as np
import csv
#np.set_printoptions(threshold=np.inf)
# from google.colab import drive 
# !gdown --id '1wNKsrdAxQ29G15kgpBy_asjTcZRRgmsCZRm' --output data.zip
# !unzip data.zip
# data = pd.read_csv('gdrive/My Drive/hw1-regression/train.csv', header = None, encoding = 'big5')
num = 9
data = pd.read_csv('./train.csv', encoding = 'big5')
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
        sample[0]=np.zeros(480);
        sample[8]=np.zeros(480);
        #sample[2]=np.zeros(480);
        sample[12]=np.zeros(480);
        sample[17]=np.zeros(480);

    month_data[month] = sample
    #print(np.shape(sample))
    #print("sample = ",sample)

x = np.empty([12 * 471, 18 * num], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + num].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + num] #value
#print(x)
#print(y)
import math
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]
x = x[: math.floor(len(x) * 0.8), :]
y = y[: math.floor(len(y) * 0.8), :]

mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
print(std_x)
mean_x_v = np.mean(x_validation, axis = 0) #18 * 9 
std_x_v = np.std(x_validation, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
for i in range(len(x_validation)):#12 * 471
    for j in range(len(x_validation[0])): #18 * 9 
        if std_x_v[j] != 0:
            x_validation[i][j] = (x_validation[i][j] - mean_x_v[j]) / std_x_v[j]
dim = 18 * num + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([math.floor(12 * 471*0.8), 1]), x), axis = 1).astype(float)
x_validation = np.concatenate((np.ones([1131, 1]), x_validation), axis = 1).astype(float)
learning_rate = 0.5401
iter_time = 10000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
train_loss=[]
dev_loss=[]
for t in range(iter_time):
    t_loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
    d_loss = np.sqrt(np.sum(np.power(np.dot(x_validation, w) - y_validation, 2))/1131)
    train_loss.append(t_loss)
    dev_loss.append(d_loss)
    if(t%100==0):
        print(str(t) + ":" + str(t_loss))
        print(str(t) + ":" + str(d_loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)

np.save('weight_best.npy', w)
np.save('std_x_best.npy',std_x)
np.save('mean_x_best.npy',mean_x)
testdata = pd.read_csv(sys.argv[1], header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype = float)
test_y = np.empty([240*9,1], dtype = float)

for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
    #print(test_y[:,0])

for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)

#print(np.sqrt(np.sum(np.power((ans_y - y_validation), 2))/1131))#rmse

with open(sys.argv[2], mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    #print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        #print(row)