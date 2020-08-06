# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Feature Scaling (Normalization here)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

# Training the SOM (Using the already created file minisom.py)
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len=15, sigma= 1.0, learning_rate=0.5)
## 10 * 10 grid of neurons each having a weight vector of 15 elements 
## input len = no. of columns in X, sigma = radius
som.random_weights_init(X)  # initialize the weights
som.train_random(data = X, num_iteration=100) # train them

# Visualizing the results

## MID of a specific winning node is the mean of the distances of all the neurons around the winnind node inside the neighbourhood that we defined using sigma.
## Higher the MID, then more the winning node will be far away from its neighbpurs inside the neighbourhood
## Higher the MID, the more the winning node is the outlier
## majority of the winning nodes define the rules to be followed, so an outlying neuron far from this majority of neurons is therofore far from the general rules.
## Thants how we'll get the frauds since, we will calculate the MID for each neuron so we will simply take the winning node that has the highest MID

## We will use colors for the winning nodes such that the higher the MID< the closer to white the color will be.

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T) # Provides the MID of all the winning nodes
colorbar() # To add the range
## highest MID is white and lowest is white
## Now we can mark if those who are fraud got the approval or not
## Red who didnt get approval and gree who got
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):# iis index, x = vecotrs of customers
    w = som.winner(x) # returns the winning node of customer x
    plot(w[0] + 0.5, 
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2) # to put the marker in the middle of the square and put the correct marker using y[i]
show()

# Finding the frauds
mappings = som.win_map(X)  # MApping of winning node with customers
fruads =  mappings[(3,9)]
frauds = sc.inverse_transform(fruads) 

