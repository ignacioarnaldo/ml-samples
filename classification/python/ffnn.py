# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 19:08:34 2015

@author: nacho
"""

import neurolab as nl
import pandas as pd
import numpy as np
from sklearn import preprocessing

#### READ TRAIN DATA ####
train_df=pd.read_csv('../../datasets/classification/banknoteTrain.csv', sep=',',header=None)
train_data = train_df.as_matrix()
NUM_EXEMPLARS = train_data.shape[0]
NUM_FEATURES = train_data.shape[1] - 1

train_features = train_data[:, 0:NUM_FEATURES]
scaler_features = preprocessing.MinMaxScaler(feature_range=(-1, 1))
norm_train_features = scaler_features.fit_transform(train_features)

train_targets = train_data[:,NUM_FEATURES]
scaler_targets = preprocessing.MinMaxScaler(feature_range=(-1, 1))
norm_train_targets = scaler_targets.fit_transform(train_targets)
norm_train_targets = norm_train_targets.reshape(NUM_EXEMPLARS,1)

# Create network with 2 layers and random initialized
minmax = []
for i in range(0, NUM_FEATURES):
      minmax.insert(i,[min(train_features[:,i]),max(train_features[:,i])])  

net = nl.net.newff(minmax , [10,1])
num_layers = len(net.layers)
for i in range(0, num_layers):
    nl.init.initwb_reg(net.layers[i])


#### TRAIN NN ####
#error = net.train_gd(train_features, train_targets, epochs=500, goal=0.01)
nl.train.train_bfgs(net,train_features, norm_train_targets,epochs=1000,show=10,goal=0.0000001) 


#### READ TEST DATA ####
test_df=pd.read_csv('../../datasets/classification/banknoteTest.csv', sep=',',header=None)
test_data = test_df.as_matrix()
test_features = test_data[:, 0:NUM_FEATURES]
norm_test_features = scaler_features.fit_transform(test_features)
test_targets = test_data[:,NUM_FEATURES]


#### PREDICT ####

# Simulate network
norm_preds = net.sim(norm_test_features)
preds = scaler_targets.inverse_transform(norm_preds)
preds = norm_preds[:,0]
preds[preds>=0.5] = 1
preds[preds<0.5] = 0

#### PERFORMANCE METRICS ####
conf_matrix = pd.crosstab(preds,test_targets, rownames=["Pred"], colnames=["Actual"])
print conf_matrix

accuracy = np.sum(preds==test_targets) / float(len(test_targets))
print accuracy


