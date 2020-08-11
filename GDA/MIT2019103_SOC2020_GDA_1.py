import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

def transform(x, mean, det, inv_sigma):
    cols = x.shape[0]
    c = 1 / ((2*np.pi)**(cols/2) * det)
    res = c * np.exp(-0.5 * np.dot(np.dot((x - mean), inv_sigma), (x - mean).T))
    return res

def box_muller(col1, col2):
    box_1 = np.sqrt(-2 * np.log(col1)) * np.cos(2 * np.pi * col2)
    box_2 = np.sqrt(-2 * np.log(col1)) * np.sin(2 * np.pi * col2)
    return box_1,box_2

df = pd.read_csv('microchip.csv')
X = df.iloc[:,:-1].values
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = df.iloc[:, -1].values

median = np.mean(X[:,0])
index = np.where(X[:,0] == 0)[0][0]
X[index][0] = median

median = np.mean(X[:,1])
index = np.where(X[:,1] == 0)[0][0]
X[index][1] = median

box1, box2 = box_muller(X[:,0], X[:,1])
X_box = np.array(list(zip(box1,box2)))
X_box = X_box[np.invert(np.isinf(X_box).any(1))]
X_box = np.nan_to_num(X_box)
X_train, X_test, y_train, y_test = train_test_split(X_box,y, test_size = 0.3    , random_state = 11)

y_mean = y.mean()
pos_data = X_train[y_train == 1]
neg_data = X_train[y_train == 0]

pos_mean = pos_data.mean(axis = 0)
neg_mean = neg_data.mean(axis = 0)

pos_data = pos_data - pos_mean
neg_data = neg_data - neg_mean

sigma = ((pos_data.T @ pos_data) + (neg_data.T @ neg_data))/X.shape[0]
inv = np.linalg.inv(sigma)
det = np.sqrt(np.linalg.det(sigma))
#testing
y_pred = []

for row in X_test:
    pos_prob = transform(row, pos_mean, det, inv) * y_mean
    neg_prob = transform(row, neg_mean, det, inv) * (1 - y_mean)
    
    if pos_prob > neg_prob:
        y_pred.append(1)
    else:
        y_pred.append(0)
        
print(accuracy_score(y_test, y_pred))