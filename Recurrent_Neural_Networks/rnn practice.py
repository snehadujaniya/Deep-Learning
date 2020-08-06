# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:07:27 2020

@author: ABC
"""

# Recurrent Neural Network



# Part 1 - Data Preprocessing

#1 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2 Importing the training set
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:,1:2].values 
  ### Because we need only the second column, .values is added to make it an array

#3 Feature Scaling - Standardization (x-mean/sigma), Normalization (x-min/(max-min))
  
  ### it is advised that whenever we have sigmoid as the activation function in the output layer
  ### in RNN, we must use Normalization

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1)) # the range in which we need the values to be(default is 0,1)
training_set_scaled = sc.fit_transform(training_set)

#4 Creating a data structure with 60 timesteps and 1 output
 ### 60 timesteps that it will use at each step to predict the following timestep
X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0]) # we need the 60 timesteps before the ith financial day
    # 0 abive is for the only column that we have
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

#5 Reshaping - Adding some more dimensions so we can predict what we want
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # see keras documentation -> Recurrent layeers -> input shape
  ### 3D tensor with shape (batch_size, timesteps, input_dim)
  ### Change axis in X_train and see the different dimensions

# Part 2 - Building the RNN

#1 Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#2 Initializing the RNN
regressor = Sequential()

#3 Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape = (X_train.shape[1],1))) # Number of LSTM cells/neurons, Return sequence is True because if we add
 ### another LSTM layer after this one, it must be ture for this one. Only the last layer has it equal to False.
 ### input_shape = shape of the input contained in the X_train which is 3D. byt we only have to include the last two observations. 1st is already taken into account 
regressor.add(Dropout(.2))

#4 Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#5 Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(.2))

#6 Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#7 Adding the output layer
regressor.add(Dense(units = 1))

#8 Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

#9 Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs=100, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

#1 Getting the real stock price of 2017
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:,1:2].values 

#2 Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)  ## This is because we havent use iloc to get the inputs. therefore, we use this

inputs = sc.transform(inputs)

X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])

X_test= np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#3 Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Goolge Stock price Prediction')
plt.xlabel('Time')
plt.ylabel('Goolge Stock Price')
plt.legend()
plt.show()


## Our curve is rather smooth than with so many breaks. It surely doesnt converge like the actual one 
## but it really performs well! 
## it will be better if we have many other variables instead of having just one!


















