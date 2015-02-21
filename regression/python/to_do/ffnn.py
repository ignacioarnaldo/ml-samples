# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 19:08:34 2015

@author: nacho
"""

import neurolab as nl
import pandas as pd
import numpy as np


#### READ TRAIN DATA ####
train_df=pd.read_csv('../../datasets/regression/winequality-white-train.csv', sep=',',header=None)
train_data = train_df.as_matrix()
NUM_FEATURES = train_data.shape[1] - 1
train_features = train_data[:, 0:NUM_FEATURES-1]
train_targets = train_data[:,NUM_FEATURES]

# Create network with 2 layers and random initialized
net = nl.net.newff([[-7, 7]],[5, 1])

#### TRAIN NN ####
error = net.train(train_features, train_targets, epochs=500, show=100, goal=0.02)


#### READ TEST DATA ####
test_df=pd.read_csv('../../datasets/regression/winequality-white-test.csv', sep=',',header=None)
test_data = test_df.as_matrix()
test_features = test_data[:, 0:NUM_FEATURES-1]
test_targets = test_data[:,NUM_FEATURES]


#### PREDICT ####
preds = net.sim(test_features)
# Simulate network


#### PERFORMANCE METRICS ####
mae = np.sum(np.abs(preds-test_targets)) / float(len(test_targets))
mse = np.sum(np.square(preds-test_targets)) / float(len(test_targets))
print '\nMAE:\t'+  str(mae)
print '\nMSE:\t'+  str(mse)