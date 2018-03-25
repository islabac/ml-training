#! /usr/local/bin/python3

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_predict

# Download iris dataset
iris = load_iris()

# Set predictor and outcome Sepal Length, Sepal Width
features, y = iris.data.T, iris.target

# Select x min, max for Sepal length feature
x_min, x_max = features[0].min() - .5, features[0].max() + .5
# Select y min, max for Sepal width feature
y_min, y_max = features[1].min() - .5, features[1].max() + .5

# Store axe stack for creating figure
plt.figure(1, figsize=(8, 6))
# Clear stack
plt.clf()
# Plot the training points
plt.scatter(features[0], features[1], c=y, cmap=plt.cm.Set1, edgecolor='k')

# Set x,y label
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# Select x min, max for Sepal length feature
a_min, a_max = features[2].min() - .5, features[2].max() + .5
# Select y min, max for Sepal width feature
b_min, b_max = features[3].min() - .5, features[3].max() + .5

# Store axe stack for creating figure
plt.figure(2, figsize=(8, 6))
# Clear stack
plt.clf()
# Plot the training points
plt.scatter(features[2], features[3], c=y, cmap=plt.cm.Set1, edgecolor='k')

# Set x,y label
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])

plt.xlim(a_min, a_max)
plt.ylim(b_min, b_max)
plt.xticks(())
plt.yticks(())

plt.show()
