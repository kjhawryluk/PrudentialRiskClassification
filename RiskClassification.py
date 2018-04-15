# coding: utf-8
from __future__ import division
import matplotlib as mpl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.utils import shuffle
import datetime
import sklearn.metrics
from sklearn.model_selection import KFold

# get_ipython().magic(u'matplotlib inline')
mpl.rc('figure', figsize=[12, 8])  # set the default figure size
import itertools, random, math
df = pd.read_csv('./train.csv', na_values="?")
print(df.head())


kf = KFold(n_splits=5) # Define the split - into 2 folds
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
KFold(n_splits=2, random_state=None, shuffle=False)
for train_index, test_index in kf.split(X):
 print(“TRAIN:”, train_index, “TEST:”, test_index)
 X_train, X_test = X[train_index], X[test_index]
 y_train, y_test = y[train_index], y[test_index]
('TRAIN:', array([2, 3]), 'TEST:', array([0, 1]))
('TRAIN:', array([0, 1]), 'TEST:', array([2, 3]))