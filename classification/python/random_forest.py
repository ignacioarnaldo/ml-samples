from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

#### READ TRAIN DATA ####
#NUM_FEATURES = 4
#train_df=pd.read_csv('/media/DATA/datasets/banknoteAuthentication/banknoteTrain.csv', sep=',',header=None)

NUM_FEATURES = 200
train_df=pd.read_csv('/home/nacho/experiments/2014-LanguageData/5classes/data/200/train_ld_200.csv', sep=',',header=None)

train_data = train_df.as_matrix()
train_features = train_data[:, 0:NUM_FEATURES-1]
train_targets = train_data[:,NUM_FEATURES]

#### TRAIN RANDOM FOREST ####
clf = RandomForestClassifier(n_estimators=10,n_jobs=8)
clf.fit(train_features, train_targets)


#### READ TEST DATA ####
#test_df=pd.read_csv('/media/DATA/datasets/banknoteAuthentication/banknoteTest.csv', sep=',',header=None)
test_df=pd.read_csv('/home/nacho/experiments/2014-LanguageData/5classes/data/200/test_ld_200.csv', sep=',',header=None)
test_data = test_df.as_matrix()
test_features = test_data[:, 0:NUM_FEATURES-1]
test_targets = test_data[:,NUM_FEATURES]

#### PREDICT ####
preds = clf.predict(test_features) 
print clf.feature_importances_

#### PERFORMANCE METRICS ####
conf_matrix = pd.crosstab(preds,test_targets, rownames=["Pred"], colnames=["Actual"])
print conf_matrix

accuracy = np.sum(preds==test_targets) / float(len(test_targets))
print accuracy