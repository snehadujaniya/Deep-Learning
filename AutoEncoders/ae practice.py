# AutoEncoders

# Importing the libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
## We are going to do inheritance here..we will inherit a modeule from the PyTorch library
class SAE(nn.Module): # Inherited module whose name is Module
    def __init__(self, ):
        super(SAE, self).__init__()
        ## first full connection between input and first hidden layer
        ## We will take self to represent the object of the class
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20,10)
        self.fc3 = nn.Linear(10,20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))  # here we are passing the input vector throgh the activation function so it gets encoded to a smaller vector
        ## therefore, x becoems the new encoded vector
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)  # final decoding so we dont apply the activation layer
        return x
    
sae = SAE()
criterion = nn.MSELoss()  # From the nn module # stochastoc gradient basically
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay = 0.5) # first we have to enter all the parameters of our auto encoders, learning rate is second, third is the decay - used to reduce the learning rate to get convergence

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    training_loss = 0
    s = 0. # will count the number of users who rated at least any movies so we can keep track of users who didnt and then dont include them in our calculations
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0) # we will add a new dimension as it doesnt accept a simgle vector as input
        target = input.clone()
        if torch.sum(target.data > 0) > 0:  # this means that the user rated atleast one movie
            output = sae(input)  # we get our first output vector, output is the vector of predicted ratings
            target.require_grad = False  # when we calculate stohastoc gradient descent, we need to calculate it only on input and not on target since both are same so we try to reduce the calculations
            output[target == 0] = 0  # these values dont count while calculations
            loss = criterion(output, target) #predicted ratings and real ratings
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # to make the denominator never equal to zero to avoid infinite loop. so adding a very small value that doesn't create any bias # why do we need this? represents the average of the error by only considering the movies that were rated because we only considered the movies that got non-zero ratings # that is why we take mean only for those that were chosen initially
            loss.backward()
            train_loss += np.sqrt(loss.item()*mean_corrector)
            s += 1.
            optimizer.step() # to update the weoghts; backward decide the direction to which the weights will be changed either inc or dec and optimizer step decides the intensity/ amount with which it will be increased or decreased
    print('epoch: '+str(epoch)+ ' loss: '+str(train_loss/s))
        
# loss represents the average of the difference between the real and predicted ratings

#### now we just need to change the class for any other type of auto encode or maybe the no. epochs
        
test_loss = 0
s = 0.
training_loss = 0
s = 0. # will count the number of users who rated at least any movies so we can keep track of users who didnt and then dont include them in our calculations
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0) # we need the input corresponding to the user so we will keep the training set
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:  # this means that the user rated atleast one movie
        output = sae(input)  # we get our first output vector, output is the vector of predicted ratings
        target.require_grad = False  # when we calculate stohastoc gradient descent, we need to calculate it only on input and not on target since both are same so we try to reduce the calculations
        output[target == 0] = 0  # these values dont count while calculations
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # to make the denominator never equal to zero to avoid infinite loop. so adding a very small value that doesn't create any bias # why do we need this? represents the average of the error by only considering the movies that were rated because we only considered the movies that got non-zero ratings # that is why we take mean only for those that were chosen initially
        test_loss += np.sqrt(loss.item()*mean_corrector)
        s += 1.
print('loss: '+str(test_loss/s))
        
            
            
        
    
    
  
        
