import numpy as np
np.random.seed(19)

def train(X):
    W = np.random.uniform(-1,1,(100,2))
    lr = 0.1
    min_lr = 0.001
#    while lr > min_lr:
    prev_output = np.zeros(X.shape[0])
    for epochs in range(100):
        count = 0
        for i in range(X.shape[0]) : 
            d = X[i] - W
            out = np.argmin(np.linalg.norm(d, axis = 1))
            if epochs != 0:
                count += 1 if prev_output[i] != out else 0
            prev_output[i] = out
            W[out] += lr*(X[i] - W[out])
#        lr -= 0.1*lr
#        print(epochs, count)
        if epochs != 0 and count == 0:
            break
    print("epochs : ",epochs)
    return W
    
X = np.random.uniform(-1,1,(1500,2))
X_test = np.array([[0.1,0.8], [0.5, -0.2], [-0.8, -0.9], [-0.6, 0.9]])

W = train(X)

for i in range(X_test.shape[0]):
    d = X_test[i] - W
    j = np.argmin(np.linalg.norm(d, axis = 1))
    print(X_test[i],j)