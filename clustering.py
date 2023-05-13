#-------------------------------------------------------------------------
# AUTHOR: Austin Celestino
# FILENAME: clustering.py
# SPECIFICATION: Does clustering on a dataset
# FOR: CS 4210- Assignment #5
# TIME SPENT: 3 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics
import csv

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = np.array(df.values)[:,:64]
X_points = []
y_points = []
kmeans = KMeans(n_clusters = 0, random_state=0)

#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code

for k in range(21):
    if 2 <= k <= 20:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
        score = silhouette_score(X_training, kmeans.labels_)
        X_points.append(k)
        y_points.append(score)



#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
plt.plot(X_points, y_points)
plt.show()

#reading the test data (clusters) by using Pandas library
#--> add your Python code here
testdf = pd.read_csv('testing_data.csv', sep=',', header=None)
with open('testing_data.csv', 'r') as csvfile:
    for i, line in enumerate(csvfile):
        pass
#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here

labels = np.array(testdf.values).reshape(1, (i + 1))[0]

#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here
