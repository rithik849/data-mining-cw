from numpy.lib.function_base import trim_zeros
import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import requests
import numpy as np
import math
# Part 3: Mining text data.

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_3(data_file):
	df = pd.read_csv("data/"+data_file,encoding='latin-1')
	return df

# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
	return list(set(df['Sentiment']))

# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
	a = df['Sentiment'].value_counts()
	return a.index[1]

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
	groups = df.groupby('Sentiment')
	sub_group = groups.get_group('Extremely Positive')
	return sub_group['TweetAt'].mode()[0]

# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.strip().str.lower()
	return df

# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.strip().str.replace(r'[^a-zA-Z ]',' ',regex=True)
	return df

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.strip().str.replace(r'[ ]{2,}',' ',regex=True)
	return df

# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.strip().str.split(' ')
	return df

# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
	# Without removing stop words and without stemming: 1350959
	# Removing stop words and stemming: 690672
	return sum(tdf['OriginalTweet'].apply(lambda x: len(x)))

# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
	# Without removing stop words and without stemming: 80071
	# Removing stop words and stemming: 68766
	results = Counter()
	tdf['OriginalTweet'].apply(results.update)
	return len(results) 

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
	# Without removing stop words and without stemming: ['the', 'to', 't', 'co', 'and', 'https', 'covid', 'of', 'a', 'in']
	# Removing stop words and stemming: ['http', 'covid', 'coronaviru', 'price', 'store', 'supermarket', 'food', 'groceri', 'peopl', 'consum']
	results = Counter()
	tdf['OriginalTweet'].apply(results.update)
	# Sorted by most to least frequent
	sortedWords = sorted(results.items(), key = lambda x: -x[1])
	return [sortedWords[i][0] for i in range(k)]

def plot_histogram(tdf,num=10):
	freqAnalysis = tdf.copy()
	num_corpus = len(tdf)
	freqAnalysis['OriginalTweet'] = freqAnalysis['OriginalTweet'].apply(lambda x:set(x))
	results = Counter()
	freqAnalysis['OriginalTweet'].apply(results.update)
	most_freq_items = sorted(results.items(),key= lambda x : x[1])
	most_freq_words , freq_count = list(zip(*most_freq_items))

	freq_count = list(np.array(freq_count)/float(num_corpus))
	plt.figure()
	plt.plot(most_freq_words[-num:],freq_count[-num:])
	plt.show()
	
	return most_freq_words[-10:]

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
	link = "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt"
	
	stop_words = requests.get( link ).content.decode('utf-8').split( "\n" )
	stop_words = set(stop_words)

	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda x: [token for token in x if len(token)>2])
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda x: [token for token in x if token not in stop_words])
	return tdf

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
	stemmer = PorterStemmer()
	#SnowballStemmer("porter")
	# Find all unique words
	results = Counter()
	tdf['OriginalTweet'].apply(results.update)
	unique_words = list(results.keys())
	# Map each unique word to its stem
	stemmed_words = [stemmer.stem(word) for word in unique_words]
	stem_dict = dict(zip(unique_words,stemmed_words))
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda x : [stem_dict[token] for token in x])
	return tdf

# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
	s = time.time()
	lower_case(df)
	remove_non_alphabetic_chars(df)
	remove_multiple_consecutive_whitespaces(df)
	tokenize(df)
	remove_stop_words(df)
	stemming(df)
	e = time.time()
	print(str(e-s)+'s')

	df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: ' '.join(x))
	arr = np.array(df['OriginalTweet'])
	labels = np.array(df['Sentiment'])

	# ngram_range (min,max)
	vectorized = CountVectorizer(analyzer='word')
	# Hyper-params
	vectorized.ngram_range=(1,4)
	vectorized.min_df = 1
	vectorized.max_df = 300

	vocab = vectorized.fit_transform(arr)

	clf = MultinomialNB()
	clf.fit(vocab,labels)
	pred = clf.predict(vocab)
	return pred

# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
	score = accuracy_score(y_pred,y_true)
	return round(score*1000)/1000



a = read_csv_3('coronavirus_tweets.csv')
s = time.time()
t = mnb_predict(a)
e = time.time()
print(str(e-s)+'s')
print(mnb_accuracy(t,a['Sentiment']))



