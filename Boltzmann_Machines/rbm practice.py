# Boltzmann Machines In this one we are predicting whether a person will like the movie or not!

#1 Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn  # to implement Neural nets in pytorch
import torch.nn.parallel # parallel computation
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#2 Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') #engine is to make sure everything gets open efficiently
   # this is just to show us the movies. won't be used
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

#3 Preparing the training anf the test set
   # Here we will not go for 5 fold cross validation therefore, we import only 1 test train split
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t') # this is 80% of the actual dat set of 100k observations.
   # tensors are being used to convert to ARRAY
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t') 
test_set = np.array(test_set, dtype = 'int')

#4 Getting the number of users and movies (So we can make a matrix where rows = users, columns = movies and cells will contain the rating)
   # if the user didn't vote, tere will be a zero there
   # size of two matrix will be equal (same rwos and columns)

nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

#5 Converting the data into an array with users in lines and movies in columns
def convert(data):
    # we wont create a 2D array. We will create a list of lists because torch expects this
    # the main list will have 943 lists because there are 943 users and each 943 lists will have 1682 elements which is the no. of movies
    new_data = []
    for id_users in range(1, nb_users + 1): #because upper bound is excluded
        id_movies = data[:,1][data[:,0] == id_users] # list of all movies rated by this user
        id_ratings = data[:,2][data[:,0] == id_users] # list of ratings of that user
        # this is the one that he rated onlu but we need all
        # so we initialise a listof zeroes and then replace the ones he rated
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)


#6 Converting the data into the torch tensors
   # Tensors are arrays that contain elements of a single data type. they are multidimensional matrix but instead of being a numpy array, it is a pytorch array
   # with tensorflow we use tensors which are tensorflow tensors
   # we have to convert our training and test set to torch tensors
   # two different multidimensional tensors based of pytorch

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)  # not visible in variable explorer but its there

#7 Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
    # Why in binary format? so that the output of rbm is consistent to already present data
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


#8 Creating the architecture of the Neural Network

    # Probabilistic graphical model
    # we will define some functions
    # 1st funct: rbm object that will be created afterwards
    # 2nd funct: sample h - that will sample the probabilities of hidden nodes given the visible nodes
    # 3rd funct: sample v - that will sample the probabilities of visible nodes given the hidden nodes
    
    # 1st function
class RBM():
    # we always have to start with init function that defines the parameteres of the object that will be created once the class is made
    def __init__(self, nv, nh): # no. of hidden and visible nodes
        # in init, we keep those parameters that we need for the model that are weighs and biases. so we initialize them in this func
        self.W = torch.randn(nh, nv) #initializes a tensor of size nh, nv acc to normal distribution mean 0 and variance 1
        self.a = torch.randn(1, nh) # first dimension corres to batch and second to the bias, a = bias for prob of hidden nodes given visible nodes 
        self.b = torch.randn(1, nv)
        
    def sample_h(self, x): 
        # this is nothing but sigmoid activation function which is product of X and weights + a
        wx = torch.mm(x, self.W.t())
        # whats inside the activation function is a linear function of neurons where W is the coefficient and a is the constant
        activation = wx + self.a.expand_as(wx) # to make sure the bias is applied to each line of the mini batch
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

            
# MCMC Markov Chain Monte Carlo effect

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))      
            
            
            
            
            
            
            
            
            