# coding: utf-8
from __future__ import division
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn import preprocessing
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import Imputer
import datetime

# get_ipython().magic(u'matplotlib inline')
mpl.rc('figure', figsize=[12, 8])  # set the default figure size
full_df = pd.read_csv('./train.csv', na_values=np.nan)
col_type = pd.read_csv('./column_type.csv')

# Change this to use more samples.
df = full_df.sample(10000)

class_column = 'Response'
id_column = 'Id'
scorer = sklearn.metrics.make_scorer(sklearn.metrics.cohen_kappa_score, weights='quadratic')
non_class_columns = [c for c in df.columns if c != class_column and c != id_column]

# Add Columns to track columns with null values
for col in non_class_columns:
	if df[col].isnull().sum() > 0:
		df[str(col) + '_nulls'] = df[col].isnull()

# Replace nulls in continuous for their medians
continuous_cols = list(col_type.loc[col_type['Data_Type'] == 'continuous', 'Column'])

continuous_imputer = Imputer(missing_values=np.nan, strategy='median', axis=0)
continuous_imputer.fit(df[continuous_cols])
df[continuous_cols] = continuous_imputer.transform(df[continuous_cols])

# Encode categorical data as numbers
label_encoder = preprocessing.LabelEncoder()
categorical_cols = list(col_type.loc[col_type['Data_Type'] == 'categorical', 'Column'])
for col in categorical_cols:
	label_encoder.fit(df[col])
	df[col] = label_encoder.transform(df[col])

# Replace nulls in continuous for their most frequent value
non_continuous_cols = list(
	col_type.loc[(col_type['Data_Type'] != 'continuous') & (col_type['Data_Type'] != 'dummy'), 'Column'])
not_continuous_imputer = Imputer(missing_values=np.nan, strategy='most_frequent', axis=0)
not_continuous_imputer.fit(df[non_continuous_cols])
df[non_continuous_cols] = not_continuous_imputer.transform(df[non_continuous_cols])

# Replace categorical (non dummy) columns with dummy columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Redefine with new dummy columns
non_class_columns = [c for c in df.columns if c != class_column and c != id_column and not ('_null' in c)]


# Plot validation and learning curves.
def plot_curve(train_scores, test_scores, param_range, title, x_axis_title, override_tics=False):
	train_scores_mean = np.mean(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	plt.clf()
	plt.title(title)
	plt.xlabel(x_axis_title)
	plt.ylabel("Quadratic Weighted Kappa Score")
	plt.ylim(0.0, 1.0)

	# Plot param range on x axis.
	if override_tics:
		plt.xticks(param_range)
	lw = 2
	plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)

	plt.plot(param_range, test_scores_mean, label="Test score", color="navy", lw=lw)

	plt.legend(loc="best")
	plt.savefig(title + ".pdf")


def get_best_depth():
	train_scores, test_scores = validation_curve(tree.DecisionTreeClassifier(),
												 df[non_class_columns], df[class_column],
												 'max_depth', range(2, 22, 2), cv=5, scoring=scorer)
	param_range = range(2, 22, 2)
	plot_curve(train_scores, test_scores, param_range, "Validation Curve For Decision Tree", 'Depth',
						  override_tics=True)
	test_scores_mean = np.mean(test_scores, axis=1)
	best_depth = (np.argmax(test_scores_mean) + 1) * 2
	return best_depth


def plot_decision_tree_learning_curve(best_depth):
	# Get learning scores for decision tree.
	train_sizes, learning_train_scores, learning_test_scores = learning_curve(
		tree.DecisionTreeClassifier(max_depth=best_depth), df[non_class_columns], df[class_column],
		train_sizes=np.linspace(0.1, 1.0, 7),
		cv=5, scoring=scorer, n_jobs=1)

	# Plot learning scores for decision tree.
	plot_curve(learning_train_scores, learning_test_scores, train_sizes, 'Sample Size', "Learning Curve For Decision Tree")


# Logistic Regression Validation Curve
def get_best_c(features, number_features_analyzed=None, n_jobs=None):
	c_range = np.linspace(1, 1000, 7)

	if number_features_analyzed:
		logistic_columns = features[0: number_features_analyzed]
	else:
		logistic_columns = features

	train_scores, test_scores = validation_curve(
		LogisticRegression(),
		df[logistic_columns], df[class_column],
		param_name='C', param_range=c_range, cv=5, scoring=scorer, n_jobs=n_jobs)

	plot_curve(train_scores, test_scores, c_range, 'Validation Curve For Logistic Regression', 'C')
	test_scores_mean = np.mean(test_scores, axis=1)
	return c_range[np.argmax(test_scores_mean)]


# plot the learning curve for the logistic regression model, using the best c value.
def plot_regression_learning_curve(c, features, number_features_analyzed=None, n_jobs=None):
	if number_features_analyzed:
		logistic_columns = features[0: number_features_analyzed]
	else:
		logistic_columns = features
	# Get learning scores for decision tree.
	train_sizes, learning_train_scores, learning_test_scores = learning_curve(
		LogisticRegression(C=c), df[logistic_columns], df[class_column],
		train_sizes=np.linspace(0.1, 1.0, 7),
		cv=5, scoring=scorer, n_jobs=n_jobs)

	# Plot learning scores for decision tree.
	plot_curve(learning_train_scores, learning_test_scores, train_sizes,
						  "Learning Curve For Logistic Regression", 'Sample Size', override_tics=True)


if __name__ == "__main__":
	start = datetime.datetime.today()
	print(start)

	# Create decision trees and make validation and learning curves.
	tree_best_depth = get_best_depth()
	print(f'Best Depth: {tree_best_depth}')
	plot_decision_tree_learning_curve(tree_best_depth)

	# When set to None, the code uses all the features. The school computers could do this easily; however, I set
	# it to 250 when I ran it on my local machine. Depending on your machine, you may need to change this to 250.
	number_of_features_used = None #250
	# When set to -1, the code will use parallel processing for completing the logistic regressions. The school
	# computers could make use of this; however, I had to change it to 1 to run on my local machine. Depending on your
	# machine, you may need to change this to 1.
	n_jobs = -1

	# Run logistic regressions and make validation and learning curves.
	best_c = get_best_c(non_class_columns, number_features_analyzed=number_of_features_used, n_jobs=n_jobs)
	print(f'Best C: {best_c}')
	plot_regression_learning_curve(best_c, non_class_columns,
								   number_features_analyzed=number_of_features_used, n_jobs=n_jobs)
	print(datetime.datetime.today() - start)
