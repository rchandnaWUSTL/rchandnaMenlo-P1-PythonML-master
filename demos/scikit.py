""" This module demonstrates some of the scikit-learn commands. """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

"""
# Linear Regression with scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_reg = LinearRegression()

boston = pd.read_csv('boston_housing.csv')

boston_inputs = boston[ ['LOW STATUS', 'ROOMS'] ]
boston_targets = boston['MEDIAN VALUE']

# Train the weights
lin_reg.fit(boston_inputs,boston_targets)

# Generate outputs / Make Predictions
boston_outputs = lin_reg.predict(boston_inputs)

# What's our error?
boston_mse = mean_squared_error(boston_targets, boston_outputs)

print("Square error using LOW STATUS and ROOMS (scikit way): " + str(boston_mse*len(boston)))

# Linear Regression the numpy way, for comparison:

inputs = boston.as_matrix(columns=['LOW STATUS', 'ROOMS'])
inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
targets = boston.as_matrix(columns=['MEDIAN VALUE'])

weights = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(inputs),inputs)),np.transpose(inputs)),targets)
outputs = np.dot(inputs,weights)
error = np.sum((targets-outputs)**2)

print("Square error using LOW STATUS and ROOMS (numpy way): " +str(error))
"""

"""
# First, let's load the Iris dataset through scikit
from sklearn import datasets

print("Loading Iris dataset")
iris = datasets.load_iris()
print(iris.keys())
print(iris["DESCR"])

iris_inputs = iris["data"]
iris_targets = (iris["target"]==2).astype(np.int) # 1 if Iris-Virginica, 0 else
print("Let's check that the inputs look right:\n" + str(iris_inputs[:5]))
print("Let's check that the targets look right:\n" + str(iris_targets[:5]))

# Now let's do some # Logistic Classification with Scikit-learn
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
# Train the Logisitic classifier
print("Training a logistic classifier to classify Iris-Virginica using all four inputs.")
log_reg.fit(iris_inputs, iris_targets)
iris_outputs = log_reg.predict(iris_inputs)
# Classification error metrics:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
print("Mean accuracy:")
print(log_reg.score(iris_inputs, iris_targets))

print("Confusion Matrix:")
print(confusion_matrix(iris_targets, iris_outputs))

# Precision = TP / (TP + FP) - How accurate are the positive predictions?
print("Precision (How accurate are the positive predictions?):")
print(precision_score(iris_targets, iris_outputs))

# Recall = TP / (TP + FN) - How correctly are positives predicted?
print("Recall (How correctly are positives predicted?):")
print(recall_score(iris_targets, iris_outputs))


# What about our buddy the Perceptron?
from sklearn.linear_model import Perceptron
percey = Perceptron()
print("Training a Perceptron to classify Iris-Virginica using all four inputs.")
percey.fit(iris_inputs, iris_targets)
percey_outputs = percey.predict(iris_inputs)

print("Mean accuracy:")
print(percey.score(iris_inputs, iris_targets))

print("Confusion Matrix:")
print(confusion_matrix(iris_targets, percey_outputs))

# Precision = TP / (TP + FP) - How accurate are the positive predictions?
print("Precision (How accurate are the positive predictions?):")
print(precision_score(iris_targets, percey_outputs))

# Recall = TP / (TP + FN) - How correctly are positives predicted?
print("Recall (How correctly are positives predicted?):")
print(recall_score(iris_targets, percey_outputs))
"""



# MNIST data set of handwritten digits
mnist_train = pd.read_csv("mnist_train.csv",index_col=False,header=None)
mnist_test = pd.read_csv("mnist_test.csv",index_col=False,header=None)

mnist_train_targets = mnist_train[0]
mnist_train_inputs = mnist_train[mnist_train.columns[1:]]
print("Shape of MNIST training input data set: " + str(mnist_train_inputs.shape))

mnist_test_targets = mnist_test[0]
mnist_test_inputs = mnist_test[mnist_test.columns[1:]]
print("Shape of MNIST test input data set: " + str(mnist_test_inputs.shape))

# Let's look at one of these data rows:
# some_digit = mnist_train_inputs[36001:36002]
# some_digit_image = np.array(some_digit).reshape(28,28)
# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()

# Now let's shuffle the training set to reduce bias opportunities
from sklearn.utils import shuffle
mnist_train_targets, mnist_train_inputs = shuffle(mnist_train_targets, mnist_train_inputs, random_state=42)

# Let's try our Logistic Classifier on the MNIST data, predicting digit 5
from sklearn.linear_model import LogisticRegression

log_reg_mnist = LogisticRegression()
mnist_train_targets_5 = (mnist_train_targets==5)
mnist_test_targets_5 = (mnist_test_targets==5)
"""
log_reg_mnist.fit(mnist_train_inputs, mnist_train_targets_5)
mnist_train_outputs = log_reg_mnist.predict(mnist_train_inputs)
# Classification error metrics:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
print("Training Logistic Classification of 5s")
print("Mean accuracy:")
print(log_reg_mnist.score(mnist_train_inputs, mnist_train_targets_5))

print("Confusion Matrix:")
print(confusion_matrix(mnist_train_targets_5, mnist_train_outputs))

# Precision = TP / (TP + FP) - How accurate are the positive predictions?
print("Precision (How accurate are the positive predictions?):")
print(precision_score(mnist_train_targets_5, mnist_train_outputs))

# Recall = TP / (TP + FN) - How correctly are positives predicted?
print("Recall (How correctly are positives predicted?):")
print(recall_score(mnist_train_targets_5, mnist_train_outputs))

# But how does it perform on the test set?
print()
print("And on the test set...")
mnist_test_outputs = log_reg_mnist.predict(mnist_test_inputs)
print("Mean accuracy:")
print(log_reg_mnist.score(mnist_test_inputs, mnist_test_targets_5))

print("Confusion Matrix:")
print(confusion_matrix(mnist_test_targets_5, mnist_test_outputs))

# Precision = TP / (TP + FP) - How accurate are the positive predictions?
print("Precision (How accurate are the positive predictions?):")
print(precision_score(mnist_test_targets_5, mnist_test_outputs))

# Recall = TP / (TP + FN) - How correctly are positives predicted?
print("Recall (How correctly are positives predicted?):")
print(recall_score(mnist_test_targets_5, mnist_test_outputs))
"""
# Stochastic Gradient Descent is faster on large data sets, so lets try it on a logistic function
from sklearn.linear_model import SGDClassifier
sgd_log = SGDClassifier(loss="log") # Tell it to use the log loss, or cross-entropy error
sgd_log.fit(mnist_train_inputs, mnist_train_targets_5)
sgd_log_outputs = sgd_log.predict(mnist_train_inputs)
print()
print("SGDClassifier with log loss looking for 5s:")
print("Mean accuracy on training set:")
print(sgd_log.score(mnist_train_inputs, mnist_train_targets_5))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
print("Confusion Matrix:")
print(confusion_matrix(mnist_train_targets_5, sgd_log_outputs))

print("Mean accuracy on test set:")
print(sgd_log.score(mnist_test_inputs, mnist_test_targets_5))

sgd_log_test_outputs = sgd_log.predict(mnist_test_inputs)
print("Confusion Matrix:")
print(confusion_matrix(mnist_test_targets_5, sgd_log_test_outputs))

"""
# Softmax Regression or Multinomial Logistic Regression!
print("Training a Multinomial Logistic Regression classifier for ALL digits!")
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(mnist_train_inputs, mnist_train_targets)
softmax_outputs = softmax_reg.predict(mnist_train_inputs)
print("Mean accuracy:")
print(softmax_reg.score(mnist_train_inputs, mnist_train_targets))
from sklearn.metrics import confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(mnist_train_targets, softmax_outputs))

"""

# Let's explore a topic called cross validation
from sklearn.model_selection import cross_val_score, cross_val_predict

print("Cross-validation of logistic classification of 5s")
print(cross_val_score( sgd_log, mnist_train_inputs, mnist_train_targets_5, cv=4, scoring="accuracy"))
cv_outputs = cross_val_predict( sgd_log, mnist_train_inputs, mnist_train_targets_5, cv=4)
print("Confusion matrix with cross validation outputs:")
print(confusion_matrix(mnist_train_targets_5,cv_outputs))
