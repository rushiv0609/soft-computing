import numpy as np
import matplotlib.pyplot as plt

np.random.seed(4)
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
#    return sigmoid(x)*(1.0-sigmoid(x))
    return x * (1-x)



class NeuralNetwork:

    def __init__(self, layers):
            
        self.activation = sigmoid
        self.activation_prime = sigmoid_prime

        self.weights = []
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)

        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.3, epochs=10000):

        ones = np.ones((X.shape[0],1))
        X = np.concatenate((ones, X), axis=1)
        errors = []

        for k in range(epochs):
#            i = np.random.randint(X.shape[0])
            MSE = 0.0
            for i in range(X.shape[0]):
            
                a = [X[i]]
    
                for l in range(len(self.weights)):
                        dot_value = np.dot(a[l], self.weights[l])
                        activation = self.activation(dot_value)
                        a.append(activation)
                # output layer
                error = y[i] - a[-1]
                MSE += np.sum(error**2)
#                errors.append(MSE)
                deltas = [error * self.activation_prime(a[-1])]
    
                for l in range(len(a) - 2, 0, -1): 
                    deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))
    
                deltas.reverse()
    
                for i in range(len(self.weights)):
                    layer = np.atleast_2d(a[i])
                    delta = np.atleast_2d(deltas[i])
                    self.weights[i] += learning_rate * layer.T.dot(delta)
            MSE /= X.shape[0]
            if k % 1000 == 0:
                print('epochs: %s, error : %s'%(k,MSE))
            errors.append(MSE)
            if(MSE < 0.01):
                break
            
        return errors
    
    def predict(self, x): 
        a = np.hstack(([1], x))
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return int(np.round(a))
#        return a


nn = NeuralNetwork([2,2,1])
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([1, 0, 0, 1])
#y = np.array([[0,1],[1,0],[1,0],[0,1]])
errors = nn.fit(X, y, 0.3)
for e in X:
    print(e,nn.predict(e))
plt.plot(range(len(errors)), errors)