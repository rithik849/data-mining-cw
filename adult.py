import pandas as pd
from pandas import get_dummies
import numpy as np
import sklearn as sk
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold
import sklearn.tree as tree
from sklearn.metrics import accuracy_score
import math

# Part 1: Decision Trees with Categorical Attributes

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'adult.csv'.
def read_csv_1(data_file):
	df = pd.read_csv("data/"+data_file)
	df.drop('fnlwgt',axis=1,inplace=True)
	return df

# Return the number of rows in the pandas dataframe df.
def num_rows(df):
	return len(df)

# Return a list with the column names in the pandas dataframe df.
def column_names(df):
	return list(df.columns)

# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
	return df.isnull().sum().sum()

# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
	missing_columns = []
	for column in column_names(df):
		if df[column].isnull().sum():
			missing_columns.append(column)
	return missing_columns

# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters, by rounding to the third decimal digit,
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 0.21547%, then the function should return 0.216.
def bachelors_masters_percentage(df):
	bachelors = list(df['education']).count('Bachelors')
	masters = list(df['education']).count('Masters')
	total = len(list(df['education']))
	val = (bachelors + masters) / float(total)
	return math.ceil(val*1000)/1000

# Return a pandas dataframe (new copy) obtained from the pandas dataframe df 
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
	df_new = df.copy()
	return df_new.dropna()

# Return a pandas dataframe (new copy) from the pandas dataframe df 
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function should not encode the target attribute, and the function's output
# should not contain the target attribute.
def one_hot_encoding(df):
	df_new = df.copy()
	categoricalFields = [x for x,y in zip(df_new.columns,df_new.dtypes) if x!='class']
	df_new.drop('class',axis=1,inplace=True)
	for category in categoricalFields:
		encoding = get_dummies(df_new[category],prefix=category)
		df_new = pd.concat([df_new,encoding],axis=1)
		df_new.drop([category],axis=1, inplace=True)
	return df_new

# Return a pandas series (new copy), from the pandas dataframe df, 
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
	df_new = df.copy()
	labelencoder = LabelEncoder()
	labels = labelencoder.fit_transform(df_new['class'])
	series = pd.Series(labels,copy=False)
	return series

# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train,y_train):
	clf = tree.DecisionTreeClassifier(random_state=0)
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_train)
	return pd.Series(y_pred)

# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
	equivalance = (y_pred == y_true)
	return 1 - (sum(equivalance)/float(len(y_true)))

df = read_csv_1("adult.csv")
df = data_frame_without_missing_values(df)
X_train = one_hot_encoding(df)
y_train = label_encoding(df)
pred = dt_predict(X_train,y_train)
print(dt_error_rate(pred,y_train))
