# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 18:43:05 2015

@author: Ignacio Arnaldo
"""

from sklearn import neighbors
import pandas as pd
import numpy as np


#### READ TRAIN DATA ####
train_df=pd.read_csv('../../datasets/classification/banknoteTrain.csv', sep=',',header=None)
train_data = train_df.as_matrix()
NUM_FEATURES = train_data.shape[1] - 1
print NUM_FEATURES
train_features = train_data[:, 0:NUM_FEATURES]
train_targets = train_data[:,NUM_FEATURES]


#### TRAIN RANDOM FOREST ####
#### TRAIN??? KNN ####
n_neighbors = 10
knn = neighbors.KNeighborsRegressor(n_neighbors)
clf = knn.fit(train_features,train_targets)


#### READ TEST DATA ####
test_df=pd.read_csv('../../datasets/classification/banknoteTest.csv', sep=',',header=None)
test_data = test_df.as_matrix()
test_features = test_data[:, 0:NUM_FEATURES]
test_targets = test_data[:,NUM_FEATURES]


#### PREDICT ####
preds = clf.predict(test_features)
preds[preds>=0.5] = 1
preds[preds<0.5] = 0


#### PERFORMANCE METRICS ####
conf_matrix = pd.crosstab(preds,test_targets, rownames=["Pred"], colnames=["Actual"])
print conf_matrix

accuracy = np.sum(preds==test_targets) / float(len(test_targets))
print accuracy