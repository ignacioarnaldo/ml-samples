# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 18:39:49 2015

@author: Ignacio Arnaldo
"""

from sklearn import tree
import pandas as pd
import numpy as np


#### READ TRAIN DATA ####
train_df=pd.read_csv('../../datasets/regression/winequality-white-train.csv', sep=',',header=None)
train_data = train_df.as_matrix()
NUM_FEATURES = train_data.shape[1] - 1
print NUM_FEATURES
train_features = train_data[:, 0:NUM_FEATURES-1]
train_targets = train_data[:,NUM_FEATURES]


#### TRAIN TREE ####
reg = tree.DecisionTreeRegressor()
reg.fit(train_features, train_targets)


#### READ TEST DATA ####
test_df=pd.read_csv('../../datasets/regression/winequality-white-test.csv', sep=',',header=None)
test_data = test_df.as_matrix()
test_features = test_data[:, 0:NUM_FEATURES-1]
test_targets = test_data[:,NUM_FEATURES]


#### PREDICT ####
preds = reg.predict(test_features) 


#### PERFORMANCE METRICS ####

mae = np.sum(np.abs(preds-test_targets)) / float(len(test_targets))
mse = np.sum(np.square(preds-test_targets)) / float(len(test_targets))
print '\nMAE:\t'+  str(mae)
print '\nMSE:\t'+  str(mse)