# Data Mining Coursework

Repository Structure: Data for datasets is located in the data folder. Scripts are located at the root directory.

# Adult dataset

Link : https://archive.ics.uci.edu/ml/datasets/Adult

The adult dataset is used to predict the salary of adults using categorical fields. The data is cleaned by removing rows that contain missing values and One Hot Encoding is applied to the categorical fields. This is then used to train a decision tree classifier to determine the salary class of a person.

# Wholesale Customers

Link : https://archive.ics.uci.edu/ml/datasets/

This dataset records the monetary spending on different categories of products.

We find patterns in the dataset using unsupervised learning methods such as K-means and Agglomerative Clustering. 

We compare the effects of standardizing the dataset aswell.

We evaluate the performance of the splits by using the silhouette score metric. The best k-split and clustering method is used to create a scatter plot between groupings of 2 categories.

# Coronavirus Tweets

Link : https://www.kaggle.com/datatattle/covid-19-nlp-text-classification

This dataset holds the details of various tweets and the sentiment of each tweet. The end result of processing the dataset is creating a Naive Bayesian Classifier to determine the sentiment of tweets. We remove stop words and clean the tweets by removing additional whitespaces and non-alphanumeric characters. The data is then stemmed and passed through a CountVectorizer which creates a document-term matrix. The data is then passed to a Multinomial Naive Bayesian Model that forms a prediction.
