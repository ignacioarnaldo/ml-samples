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
train_df=pd.read_csv('../../../datasets/regression/winequality-white-train.csv', sep=',',header=None)
train_data = train_df.as_matrix()
NUM_EXEMPLARS = train_data.shape[0]
NUM_FEATURES = train_data.shape[1] - 1

train_features = train_data[:, 0:NUM_FEATURES]
train_targets = train_data[:,NUM_FEATURES]
min_max_scaler = preprocessing.MinMaxScaler()
norm_train_targets = min_max_scaler.fit_transform(train_targets)

norm_train_targets = train_targets.reshape(NUM_EXEMPLARS,1)

# Create network with 2 layers and random initialized
minmax = []
for i in range(0, NUM_FEATURES):
      minmax.insert(i,[min(train_features[:,i]),max(train_features[:,i])])  

#net = nl.net.newff(minmax , [10, 1])
net = nl.net.newp(minmax , 1)

#### TRAIN NN ####
#error = net.train_gd(train_features, train_targets, epochs=500, goal=0.01)
nl.train.train_gdm(net,train_features, norm_train_targets,epochs=100,show=10) 


#### READ TEST DATA ####
test_df=pd.read_csv('../../../datasets/regression/winequality-white-test.csv', sep=',',header=None)
test_data = test_df.as_matrix()
test_features = test_data[:, 0:NUM_FEATURES]
test_targets = test_data[:,NUM_FEATURES]


#### PREDICT ####

# Simulate network
norm_preds = net.sim(test_features)

# rescale to original range
preds = min_max_scaler.inverse_transform(norm_preds)

#### PERFORMANCE METRICS ####
mae = np.sum(np.abs(preds-test_targets)) / float(len(test_targets))
mse = np.sum(np.square(preds-test_targets)) / float(len(test_targets))
print '\nMAE:\t'+  str(mae)
print '\nMSE:\t'+  str(mse)