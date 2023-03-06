import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
# Part 2: Cluster Analysis

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
	df = pd.read_csv("data/"+data_file)
	df.drop('Channel',axis=1,inplace=True)
	df.drop('Region',axis=1,inplace=True)
	return df

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
	stats = ['mean','std','min','max']
	tmp = pd.DataFrame(index = df.columns,columns = stats)
	for i in tmp.transpose():
		tmp['mean'][i] = round(df[i].mean())
		tmp['std'][i] = round(df[i].std())
		tmp['min'][i] = df[i].min()
		tmp['max'][i] = df[i].max()
	return tmp

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
	df = df.copy()
	summary = summary_statistics(df)
	return (df - summary['mean'])/summary['std']

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
def kmeans(df, k):
	kmeans = KMeans(n_clusters=k,init='random',n_init=1).fit(df)
	return pd.Series(kmeans.labels_)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
	kmeans = KMeans(n_clusters=k,init='k-means++',n_init=1).fit(df)
	return pd.Series(kmeans.labels_)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
	agglomerative = AgglomerativeClustering(n_clusters=k).fit(df)
	return pd.Series(agglomerative.labels_)

# Given a data set X and an assignment to clusters y
# return the Solhouette score of the clustering.
# Silhouetter score measures how good a cluster seperation is
# Measured as average distance of datapoint to all other points in cluster(A) - average distance of datapoint to all other points in nearest (not including current datapoint) cluster (B) / max(A,B)
def clustering_score(X,y):
	return silhouette_score(X,y)

# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
	dfReturn = pd.DataFrame(columns = ['Algorithm','data','k','Silhouette Score'])
	record = []
	for i in ["Original","Standardized"]:
		dataset=df
		if i=="Standardized":
					dataset = standardize(df)
		for j in ["Kmeans","Agglomerative"]:
			for k in [3,5,10]:
				if j=="Kmeans":
                    # Repeat k-means 10 times to account for randomness of initial points
					for l in range(10):
						result = kmeans(dataset,k)
						score = clustering_score(dataset,result)
						record.append([j,i,k,score])
				else:
					result = agglomerative(dataset,k)
					score = clustering_score(dataset,result)
					record.append([j,i,k,score])
	dfReturn = pd.DataFrame(record,columns = ['Algorithm','data','k','Silhouette Score'])
	return dfReturn

# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
	return rdf['Silhouette Score'].max()

# Run some clustering algorithm of your choice with k=3 and generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
    plt.rcParams["figure.figsize"] = (10, 9)
    k=3
    combinations = []
    colNo = len(df.columns)
    df = standardize(df)
    fig,axes = plt.subplots(nrows=5,ncols=3)
    evaluation = cluster_evaluation(df)
    # alg=Kmeans data=Original k=3 score=0.477018
    best = evaluation.iloc[evaluation['Silhouette Score'].argmax()]
    counter = 0
    for i in range(colNo-1):
        for j in range(i+1,colNo):
            # Create subsets of 2 fields
            combinations.append([df.columns[i],df.columns[j]])
            sub_df = pd.DataFrame([df[df.columns[i]],df[df.columns[j]]]).transpose()
            
            dataset = sub_df if best['data'] == 'Original' else standardize(sub_df)
            labels = agglomerative(dataset,k) if best['Algorithm'] == 'Agglomerative' else kmeans(dataset,k)
            dataset['cl'] = labels

            ax = dataset.plot(x=df.columns[i],y=df.columns[j],ax = axes[counter%5][counter//5],kind='scatter',c='cl',colormap='gist_rainbow')
            ax.set_xlabel(df.columns[i])
            ax.set_ylabel(df.columns[j])
            counter += 1
    plt.subplots_adjust(hspace=0.5)
    plt.savefig("scatter.png")
 

a = read_csv_2("wholesale_customers.csv")
scatter_plots(a)