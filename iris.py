"""iris.py"""
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np

DF = pd.read_csv('data/iris_train.csv')
pd.set_option('mode.chained_assignment', None)


# def res_to_color(res):
#         res_colors={'Iris-setosa':'orange',
# ................         'Iris-virginica':'cyan',
# ................         'Iris-versicolor':'red'
# ................        }
#      if res in res_colors:
# ........ return res_colors[res]
# .... else:
# ........ return 'grey'

# DF.plot(kind = 'scatter', x = 'petal width', y = 'petal length',
# ....c = DF['class'].apply(res_to_color))

# DF.plot(kind = 'scatter', x = 'sepal width', y = 'sepal length',
# ....c = DF['class'].apply(res_to_color))

# plt.show()

# Setosa with petals

def iris_to_target(iris):
    if iris == 'Iris-setosa':
        return 1
    else:
        return -1


TARGETS = np.reshape(DF['class'].apply(iris_to_target).values, (-1, 1))

INPUTS = DF[['petal length', 'petal width']]
INPUTS['threshold'] = -1

ERROR = 1
MIN_ERROR = 100000
MIN_ERROR_WEIGHTS = None

while ERROR != 0:

    WEIGHTS = np.random.rand(len(INPUTS.columns), 1) * 0.1 - 0.05
    OUTPUTS = np.sign(np.dot(INPUTS, WEIGHTS))
    ERROR = sum(abs(TARGETS - OUTPUTS)) // 2

    if ERROR < MIN_ERROR:
        MIN_ERROR = ERROR
        MIN_ERROR_WEIGHTS = WEIGHTS

print('Setosa with petal INPUTS:')
print(MIN_ERROR_WEIGHTS)
print(MIN_ERROR)

# Setosa with sepals

INPUTS = DF[['sepal length', 'sepal width']]
INPUTS['threshold'] = -1

ERROR = 1

COUNT = 0

MIN_ERROR = 100000
MIN_ERROR_WEIGHTS = None

while ERROR != 0 and COUNT != 1000:

# need to have fixed max times through the loop, no guarantee that ERROR will be 0

    WEIGHTS = np.random.rand(len(INPUTS.columns), 1) * 0.1 - 0.05

    OUTPUTS = np.sign(np.dot(INPUTS, WEIGHTS))

    ERROR = sum(abs(TARGETS - OUTPUTS))

    if ERROR < MIN_ERROR:
        MIN_ERROR = ERROR
        MIN_ERROR_WEIGHTS = WEIGHTS

    # print(WEIGHTS)
    # print(ERROR) #Setosa with sepal

    COUNT = COUNT + 1

print('Setosa with sepal INPUTS:')
print(MIN_ERROR_WEIGHTS)
print(MIN_ERROR)


# Virginica any two

def iris_to_target(iris):
    if iris == 'Iris-virginica':
        return 1
    else:
        return -1


TARGETS = np.reshape(DF['class'].apply(iris_to_target).values, (-1, 1))

INPUTS = DF[['sepal width', 'petal width']]

# INPUTS = DF[["petal length", "petal width"]]
# INPUTS = DF[["petal width", "sepal width"]]

INPUTS['threshold'] = -1

ERROR = 1
COUNT = 0

MIN_ERROR = 100000
MIN_ERROR_WEIGHTS = None

while COUNT != 2500 and ERROR != 0:

    WEIGHTS = np.random.rand(len(INPUTS.columns), 1) * 0.1 - 0.05

    OUTPUTS = np.sign(np.dot(INPUTS, WEIGHTS))

    ERROR = sum(abs(TARGETS - OUTPUTS))

    # print(WEIGHTS)
    # print(ERROR) #Virginica with petal

    if ERROR < MIN_ERROR:
        MIN_ERROR = ERROR
        MIN_ERROR_WEIGHTS = WEIGHTS

    COUNT += 1

print('Virginica any two:')
print(MIN_ERROR_WEIGHTS)
print(MIN_ERROR)

# Virginica all four

TARGETS = np.reshape(DF['class'].apply(iris_to_target).values, (-1, 1))

INPUTS = DF[['sepal length', 'sepal width', 'petal length',
             'petal width']]
INPUTS['threshold'] = -1

ERROR = 1
COUNT = 0

MIN_ERROR = 25000
MIN_ERROR_WEIGHTS = None

while COUNT != 5000 and ERROR != 0:

    WEIGHTS = np.random.rand(len(INPUTS.columns), 1) * 0.1 - 0.05

    OUTPUTS = np.sign(np.dot(INPUTS, WEIGHTS))

    ERROR = sum(abs(TARGETS - OUTPUTS))

    if ERROR < MIN_ERROR:
        MIN_ERROR = ERROR
        MIN_ERROR_WEIGHTS = WEIGHTS

    COUNT += 1

print('Virginica all four:')
print(MIN_ERROR_WEIGHTS)
print(MIN_ERROR)

# NOTES

# Ask what meant by competion
# Do you expect that there are two parameters that will be WAY better than all others?
    # Maybe some parameters will be better at a certain number of cycles?
