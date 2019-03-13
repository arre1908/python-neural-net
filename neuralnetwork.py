import numpy as np
import matplotlib.pyplot as mp
from pylab import show

# for testing purposes
# np.random.seed(100) 

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def sigmoid_derivative(t):
    return t * (1 - t)

class NeuralNetwork:
    def __init__(self, x, y, num_layers=2, num_nodes=2, lr=0.01, max_iter=10000):
        self.x = np.array(x)
        self.y = np.array([y]).T
        
        #create the weights from the inputs to the first layer
        self.weights = [np.random.rand(len(self.x[0]), num_nodes)]
        for i in range(num_layers-1):
            #create the random weights between internal layers
            self.weights.append(np.random.rand(num_nodes, num_nodes))
        #create weights from final layer to output node
        self.weights.append(np.random.rand(num_nodes, 1))
        
        self.outputs = np.zeros(self.y.shape)
        self.error_history = [] # for plotting
        
        self.train(lr, max_iter)

    def train(self, learning_rate, max_iter):
        for i in range(max_iter):
            self.feedforward(self.x)
            self.backprop(learning_rate)

    def feedforward(self, x):
        # contains the input, hidden, and output layers
        self.layers = [x]
        
        for w in self.weights:
            z = np.dot(self.layers[-1], w)
            self.layers.append(sigmoid(z))
        self.outputs = self.layers[-1]

    def backprop(self, learning_rate):
        error = self.outputs - self.y
        self.error_history.append(np.mean([abs(x) for x in error]))

        # output layer derivatives
        d_error = [2 * error * sigmoid_derivative(self.layers[-1])]
        d_weights = [np.dot(self.layers[-2].T, d_error[-1])]

        # hidden layer derivatives
        for index in range(len(self.weights[:-1])):
            # reverse index
            index = -(index+1)
            e = np.dot(d_error[-1], self.weights[index].T)
            d_error.append(e * sigmoid_derivative(self.layers[index-1]))
            d_weights.append(np.dot(self.layers[index-2].T, d_error[-1]))

        # change weights
        for w, d in zip(self.weights, reversed(d_weights)):
            w -= learning_rate * d

    def predict(self, inputs):
        self.feedforward(inputs)
        return np.rint(self.outputs)

    def plot_error_stats(self):
        mp.plot(range(len(self.error_history)), self.error_history)
        mp.title("Error stats")
        mp.xlabel("Iteration")
        mp.ylabel("Average Absolute Error")
        mp.ylim([0,1])
        show()

# simple test data
x = [[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]]
y = [1,0,0,1,1]
# x = [[0,0,1],[0,1,1],[1,0,1],[1,1,1]]
# y = [0,1,1,0]

# create neural network
nn = NeuralNetwork(x=x, y=y, num_layers=1, num_nodes=2, lr=.1, max_iter=10000)
nn.plot_error_stats()

# get prediction values on training set
out = nn.predict(x)
print("\nPREDICTION")
for a,b in zip(nn.outputs, out):
    print("%f --> %d" % (a, b))

# prediction accuracy -- ONLY IF Y CONTAINS THE TARGET LABELS FOR PREDICTION ABOVE
from sklearn.metrics import accuracy_score
print("Accuracy: %.2f" % (accuracy_score(y, out) * 100))
