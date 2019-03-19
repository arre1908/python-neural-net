# python-neural-net

Simple neural network classifier written in Python 3 using [numpy](https://www.numpy.org/).

Supports several tuning parameters and outputs accuracy reports & graphs to visualize convergence using [matplotlib](https://www.matplotlib.org/).

### Parameters

- `x`: 2-D array of training data
- `y`: 1-D array of target labels (0,1)
- `num_layers`: Number of hidden layers to use for the model (>= 1, default: 2)
- `num_nodes`: Number of nodes per hidden layer (>= 1, default: 2)
- `lr`: Learning rate (default: 0.01)
- `max_iter`: Number of iterations to run (default: 10,000)

### Input

Training data used in the code (source: https://stackabuse.com/creating-a-neural-network-from-scratch-in-python/) are simple examples that can illustrate the neural network's ability to fit the data and make reasonable predictions not seen in the training set.

| Person  | Smoking | Obesity | Exercise | Diabetic |
| ------- | :-----: | :-----: | :------: | :------: |
| Person1 |    0    |    1    |    0     |    1     |
| Person2 |    0    |    0    |    1     |    0     |
| Person3 |    1    |    0    |    0     |    0     |
| Person4 |    1    |    1    |    0     |    1     |
| Person5 |    1    |    1    |    1     |    1     |

Prediction test on unseen data:

| Person  | Smoking | Obesity | Exercise | Diabetic |
| ------- | :-----: | :-----: | :------: | :------: |
| Person1 |    1    |    0    |    1     |    ?     |

### Output

```text
TRAINING OUTPUTS:
0.989994 --> 1
0.019510 --> 0
0.020174 --> 0
0.987970 --> 1
0.978687 --> 1
Accuracy: 100.00%

PREDICTION
0.008062 --> 0
```

> _NOTE: This will vary when a random seed is implemented_
