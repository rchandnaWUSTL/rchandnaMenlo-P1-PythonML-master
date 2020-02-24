""" This module demonstrates decision trees. """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
#import graphviz
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

# First, let's load the Iris dataset through scikit

print("Loading Iris dataset")
iris = datasets.load_iris()

def iris_to_color(iris):
	if iris==0:
		return "red"
	elif iris==1:
		return "blue"
	else:
		return "green"

iris_inputs = iris["data"]
iris_targets = iris["target"]

iris_data = pd.DataFrame(iris["data"])
iris_type = pd.DataFrame(iris["target"])[0]

plt.scatter(iris_data[2], iris_data[3], c=iris_type.apply(iris_to_color))
plt.show()

"""
print("\n\nMultinomial logistic regression on Iris dataset")
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(iris_inputs, iris_targets)
softmax_outputs = softmax_reg.predict(iris_inputs)
print("Mean accuracy:")
print(softmax_reg.score(iris_inputs, iris_targets))
print("Confusion Matrix:")
print(confusion_matrix(iris_targets, softmax_outputs))

print("\n\nDecision Tree Classification on Iris dataset - no max depth")
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier()
tree_clf.fit(iris_inputs, iris_targets)
tree_clf_outputs = tree_clf.predict(iris_inputs)
print("Mean accuracy:")
print(tree_clf.score(iris_inputs, iris_targets))
print("Confusion Matrix:")
print(confusion_matrix(iris_targets, tree_clf_outputs))

from sklearn.tree import export_graphviz
dot_data = export_graphviz(tree_clf, out_file=None,
							feature_names=iris.feature_names,  
                         	class_names=iris.target_names,  
                         	filled=True, rounded=True,  
                         	special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render("iris")
"""
"""
print("\n\nDecision Tree Classification on Iris dataset - max depth = 2")
tree_clf2 = DecisionTreeClassifier(max_depth=2)
tree_clf2.fit(iris_inputs, iris_targets)
tree_clf2_outputs = tree_clf2.predict(iris_inputs)
print("Mean accuracy:")
print(tree_clf2.score(iris_inputs, iris_targets))
print("Confusion Matrix:")
print(confusion_matrix(iris_targets, tree_clf2_outputs))
"""
"""
print("\n\nLet's try out GridSearch!")
from sklearn.model_selection import GridSearchCV
param_grid = [ { 'max_depth': range(2,9) } ]
grid_search = GridSearchCV(tree_clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(iris_inputs, iris_targets)
print(grid_search.best_params_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
	print(mean_score,params)
"""

"""
# MNIST data set of handwritten digits
mnist_train = pd.read_csv("mnist_train.csv",index_col=False,header=None)
mnist_test = pd.read_csv("mnist_test.csv",index_col=False,header=None)
mnist_train_targets = mnist_train[0]
mnist_train_inputs = mnist_train[mnist_train.columns[1:]]
mnist_test_targets = mnist_test[0]
mnist_test_inputs = mnist_test[mnist_test.columns[1:]]
mnist_train_targets, mnist_train_inputs = shuffle(mnist_train_targets, mnist_train_inputs, random_state=42)

print("\n\nDecision Tree Classifier on MNIST!")
mnist_tree = DecisionTreeClassifier()
mnist_tree.fit(mnist_train_inputs, mnist_train_targets)
mnist_train_outputs = mnist_tree.predict(mnist_train_inputs)
print("Training mean accuracy:")
print(mnist_tree.score(mnist_train_inputs, mnist_train_targets))
print("Training confusion Matrix:")
print(confusion_matrix(mnist_train_targets, mnist_train_outputs))
mnist_test_outputs = mnist_tree.predict(mnist_test_inputs)
print("\nTest mean accuracy:")
print(mnist_tree.score(mnist_test_inputs, mnist_test_targets))
print("Test confusion Matrix:")
print(confusion_matrix(mnist_test_targets, mnist_test_outputs))
"""
