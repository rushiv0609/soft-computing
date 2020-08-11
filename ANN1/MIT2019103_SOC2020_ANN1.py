import numpy as np

np.random.seed(4)

def get_AND():
    X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    y = np.array([0,0,0,1])
    return X,y

def get_NAND():
    X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    y = np.array([1,1,1,0])
    return X,y

def get_OR():
    X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    y = np.array([0,1,1,1])
    return X,y

def get_NOR():
    X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    y = np.array([1,0,0,0])
    return X,y

def activation(x):
    return 1 if x > 0 else 0

def train(X,y, lr = 0.3):
    W = np.random.rand(3) - 0.5
    changed = True
    epoch = 1
    while changed:
        changed = False
        for i in range(X.shape[0]):
            h = np.dot(W, X[i])
            pred = activation(h)
            error = y[i] - pred
            if error != 0:
                changed = True
            W += lr*error*X[i]
        epoch+=1
    print(epoch)
    return W

def predict(X, W):
    h = np.dot(W, X)
    return activation(h)


print("AND GATE")
X, y = get_AND()
W = train(X,y,0.5)
for i in range(X.shape[0]):
    print(X[i][:-1], predict(X[i], W))    
print("\n")

print("OR GATE")
X, y = get_OR()
W = train(X,y,0.5)
for i in range(X.shape[0]):
    print(X[i][:-1], predict(X[i], W))    
print("\n")

print("NAND GATE")
X, y = get_NAND()
W = train(X,y,0.5)
for i in range(X.shape[0]):
    print(X[i][:-1], predict(X[i], W))    
print("\n")

print("NOR GATE")
X, y = get_NOR()
W = train(X,y,0.5)
for i in range(X.shape[0]):
    print(X[i][:-1], predict(X[i], W))    
print("\n")