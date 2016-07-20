# Ian Castillo Rosales 
# 07/20/2016

# What this script does: This script creates three clusters and prints the cluster centers to a file
# OUTPUT: clusterCenters.txt

# [STEP 1] To run this script, write following on terminal, and hit enter:
# 		$ PYSPARK_PYTHON=/home/cloudera/anaconda3/bin/python spark-submit sparkMLlibClustering.py

# Ian Castillo Rosales 
# 07/20/2016

# What this script does: This script creates three clusters and prints the cluster centers to a file
# OUTPUT: clusterCenters.txt

# [STEP 1] To run this script, write following on terminal, and hit enter:
# 		$ PYSPARK_PYTHON=/home/cloudera/anaconda3/bin/python spark-submit sparkMLlibClustering.py

import pandas as pd
import numpy as np
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from datetime import datetime
import sys

conf = (SparkConf()
         .setMaster("local")
         .setAppName("My app")
         .set("spark.executor.memory", "1g"))
sc 			= SparkContext(conf = conf)
sqlContext 	= SQLContext(sc)


# ===== Read files =====
# Read ad-clicks.csv file
adclicksDF = pd.read_csv('ad-clicks.csv')
adclicksDF = adclicksDF.rename(columns=lambda x: x.strip())
adclicksDF['adCount'] = 1

# Read user-session.csv file
userSessionsDF = pd.read_csv('user-session.csv')
userSessionsDF = userSessionsDF.rename(columns=lambda x: x.strip())
userSessionsDF['adCount'] = 1 

# Read users.csv file
usersDF = pd.read_csv('users.csv')
usersDF = usersDF.rename(columns=lambda x: x.strip())

# ===== Obtain age  =====
# Obtain age from user's date of birth 
now = pd.Timestamp(datetime.now())
usersDF['dob'] = pd.to_datetime(usersDF['dob'], format='%Y-%m-%d')
usersDF['dob'] = usersDF['dob'].where(usersDF['dob'] < now, usersDF['dob'] - np.timedelta64(100, 'Y'))
usersDF['Age'] = (now - usersDF['dob']).astype('<m8[Y]')


# ===== Select attributes =====
# Select user attributes for clusterings
adClicks = adclicksDF[['userId','adCount']]
Sessions = userSessionsDF[['userId','adCount']]
AgeDF = usersDF[['userId', 'Age']]


# ===== Aggregation =====
# Perform aggregation to get total ad-clicks per user
adsPerUser = adClicks.groupby('userId').sum()
adsPerUser = adsPerUser.reset_index()
adsPerUser.columns = ['userId', 'totalAdClicks']

# Perform aggregation to get total sessions per user
sessionsPerUser = Sessions.groupby('userId').sum()
sessionsPerUser = sessionsPerUser.reset_index()
sessionsPerUser.columns = ['userId', 'totalSessions']

# Merge the three tables
combinedDF = adsPerUser.merge(sessionsPerUser, on='userId')
combinedDF = combinedDF.merge(AgeDF, on='userId')

# Final training dataset (remove userId)
trainingDF = combinedDF[['totalAdClicks', 'totalSessions', 'Age']]
trainingDF.head(5)
trainingDF.shape

#Remove userId before training and keep other two attributes
sqlContext = SQLContext(sc)
pDF = sqlContext.createDataFrame(trainingDF)
parsedData = pDF.rdd.map(lambda line: np.array([line[0], line[1], line[2]]))

#Train KMeans model to create two clusters

clusters = KMeans.train(parsedData, 3, maxIterations=10, runs=10, initializationMode="random")

#redirect stdout
orig_stdout = sys.stdout
f = open('clusterCenters.txt', 'w')
sys.stdout = f

#Display the centers of two clusters
print(clusters.centers)

#Redirect back the stdout
sys.stdout = orig_stdout
f.close()