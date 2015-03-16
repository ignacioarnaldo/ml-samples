# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 18:39:49 2015

@author: Ignacio Arnaldo
"""

from sklearn.ensemble import RandomForestClassifier
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
clf = RandomForestClassifier(n_estimators=10,n_jobs=8)
clf.fit(train_features, train_targets)


#### READ TEST DATA ####
test_df=pd.read_csv('../../datasets/classification/banknoteTest.csv', sep=',',header=None)
test_data = test_df.as_matrix()
test_features = test_data[:, 0:NUM_FEATURES]
test_targets = test_data[:,NUM_FEATURES]


#### PREDICT ####
preds = clf.predict(test_features) 
print clf.feature_importances_


#### PERFORMANCE METRICS ####
conf_matrix = pd.crosstab(preds,test_targets, rownames=["Pred"], colnames=["Actual"])
print conf_matrix

accuracy = np.sum(preds==test_targets) / float(len(test_targets))
print accuracy