"""logistic.py"""
import pandas as pd
import numpy as np

DF_IRIS = pd.read_csv('data/iris_train.csv')
DF_HOUSING = pd.read_csv('data/boston_housing.csv')
DF_STOCK = pd.read_csv('data/Weekly.csv')

def logistic(weights, inputs):
    """ Computes the logistic function of the dot product
        of the weight matrix and the input matrix. """
    s = np.dot(inputs, weights)
    return 1/(1+np.exp(-1*s))

def compute_error(weights, inputs, targets):
    """ Computes the cross-entropy error for the logistic function
        assuming that targets are encoded using +1 and -1 as the
        binary values. """
    N = inputs.shape[0]
    s = np.dot(inputs, weights)
    ts = -1 * np.multiply(targets, s)
    return np.sum(np.log(1+np.exp(ts)))/N

def gradient(weights, inputs, targets):
    """ Computes the gradient of the cross-entropy error. """
    y = logistic(weights, -np.multiply(targets, inputs))
    return np.transpose(np.dot(np.transpose(y),
                               -np.multiply(targets, inputs)))

def confusion_matrix(outputs, targets):
    """ Returns a confusion matrix of Predicted versus Actual values. """
    outputs_df = pd.DataFrame(outputs, columns=['Predicted'])
    targets_df = pd.DataFrame(targets, columns=['Actual'])
    return pd.crosstab(outputs_df['Predicted'], targets_df['Actual'])

def train(inputs, targets, eta):
    """ Trains """
    max_cycles = 10000
    mag = 1
    cycles = 0
    error = 1
    weights = np.random.rand((inputs.shape[1]), 1) * eta - 0.05
    while mag > 0.1 and cycles < max_cycles:
        error = compute_error(weights, inputs, targets)
        grad = gradient(weights, inputs, targets)
        weights += -eta * grad
        mag = np.sum(grad ** 2)
        cycles += 1
    return weights, mag, error

def setosa_to_target(iris):
    """ Sets setosa as target """
    if iris == 'Iris-setosa':
        return 1
    else:
        return -1

def virginica_to_target(iris):
    """ Sets virginica as target """
    if iris == 'Iris-virginica':
        return 1
    else:
        return -1


SETOSA_TARGETS = np.reshape(DF_IRIS['class'].apply(setosa_to_target).values, (-1, 1))
VIRGINICA_TARGETS = np.reshape(DF_IRIS['class'].apply(virginica_to_target).values, (-1, 1))
INPUTS = DF_IRIS[['petal length', 'petal width', 'sepal length', 'sepal width']]

#SETOSA
print("\n")
print("Predicting whether an Iris is a Setosa Iris:")
print("Inputs: Petal Length, Petal Width, Sepal Length, Sepal Width")
WEIGHTS, _, _ = train(INPUTS, SETOSA_TARGETS, 0.001)
print("Weights:")
print(WEIGHTS)
print(confusion_matrix(np.sign(logistic(WEIGHTS, INPUTS)-0.5), SETOSA_TARGETS))
print("\n")

#VIRGINICA
print("Predicting whether an Iris is a Virginica Iris")
print("Inputs: Petal Length, Petal Width, Sepal Length, Sepal Width")
WEIGHTS, _, _ = train(INPUTS, VIRGINICA_TARGETS, 0.0009)
print("Weights:")
print(WEIGHTS)
print(confusion_matrix(np.sign(logistic(WEIGHTS, INPUTS)-0.5), VIRGINICA_TARGETS))
print("\n")

#PART 3 Predict whether an area in the Boston Housing
#data set will have a crime rate above the median

#print(DF_HOUSING.corr(method='pearson', min_periods=1))
# Columns that look significant:
# Prop Tax Rate - 0.793392
# Nox - 0.634679
# Industry - 0.590822
# Emp Distance - -0.495148
# Prior 1940 - 0.482013
# Low Status - 0.481907

TARGETS = np.reshape(np.sign(DF_HOUSING['CRIME RATE']
                             - np.median(DF_HOUSING['CRIME RATE'])), (-1, 1))
DF_HOUSING['PROP TAX RATE'] = DF_HOUSING['PROP TAX RATE'] - np.median(DF_HOUSING['PROP TAX RATE'])
INPUTS = DF_HOUSING.as_matrix(columns=['INDUSTRY', 'NOX', 'PROP TAX RATE',
                                       'EMP DISTANCE', 'PRIOR 1940'])
WEIGHTS, _, _ = train(INPUTS, TARGETS, 0.000001)

print("\n")
print("Predicting whether an area in Boston will have a crime rate above the median:")
print("Inputs: Industry, Nox, Prop Tax Rate, Emp Distance, Prior 1940")
print("Weights:")
print(WEIGHTS)
print(confusion_matrix(np.sign(logistic(WEIGHTS, INPUTS)-0.5), TARGETS))
print("\n")


#EXTRA CREDIT: STOCK MARKET

def direction_to_target(market):
    """ Sets Up as target """
    if market == 'Up':
        return 1
    else:
        return -1

TARGETS = np.reshape(DF_STOCK['Direction'].apply(direction_to_target).values, (-1, 1))
INPUTS = DF_STOCK[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]

print("Predicting the Up or Down turn of the Stock market:")
print("Inputs: Lag1, Lag2, Lag3, Lag4, Lag5, Volume")
WEIGHTS, _, _ = train(INPUTS, TARGETS, 0.0007)
print("Weights:")
print(WEIGHTS)
print(confusion_matrix(np.sign(logistic(WEIGHTS, INPUTS)-0.5), TARGETS))
