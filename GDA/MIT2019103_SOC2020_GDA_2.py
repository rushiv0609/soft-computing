import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def transform(x, mean, det, inv_sigma):
    cols = x.shape[0]
    c = 1 / ((2*np.pi)**(cols/2) * det)
    res = c * np.exp(-0.5 * np.dot(np.dot((x - mean), inv_sigma), (x - mean).T))
    return res

df = pd.read_csv('microchip.csv')
X = df.iloc[:,:-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 11)

y_mean = y.mean()
pos_data = X_train[y_train == 1]
neg_data = X_train[y_train == 0]

pos_mean = pos_data.mean(axis = 0)
neg_mean = neg_data.mean(axis = 0)

pos_data = pos_data - pos_mean
neg_data = neg_data - neg_mean

sigma = (np.dot(pos_data.T, pos_data) + np.dot(neg_data.T, neg_data))/X.shape[0]
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