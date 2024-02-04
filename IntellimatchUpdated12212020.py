# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 19:36:39 2020

@author: DanSarkar
"""

import tensorflow as ts
import pandas as pd
import seaborn as sns
import numpy as np
import pandasql as ps
import pandas as pd
import sqlite3
import pandas.io.sql as psql
import ast
import re
import datetime
import seaborn as sb
import sklearn
import statistics
import h2o
from h2o.automl import H2OAutoML
h2o.init(max_mem_size='4G')
import matplotlib.pyplot as plot
from pandasql import sqldf
from pandasql import *
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer   
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples
#import featuretools as ft
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
ps = lambda q: sqldf(q, globals())
from scipy.stats import pearsonr
sns.set(style='white', font_scale=1.2)


# sess = ts.Session()
a = ts.constant(10)
b = ts.constant(32)
print((a+b))




pysqldf = lambda q: sqldf(q, globals())

SurveyData = pd.read_csv('C:/Users/Dans-IQS/Documents/BackupfromLenovo10142020/Documents/BackupPreviousWindows/IntelliMatch/Data/SurveyDataUsedForR.csv',encoding= 'iso-8859-1')

SurveyData1=SurveyData[~SurveyData['Address_Work_Zipcode'].isnull()]

MLSData = pd.read_csv('C:/Users/Dans-IQS/Documents/BackupfromLenovo10142020/Documents/BackupPreviousWindows/IntelliMatch/Data/DataFr_flat_Cleaned_modi2.csv',encoding= 'iso-8859-1')

###

SurveyData.info()

### Create the database
con = sqlite3.connect("SurveyData1.db")

SurveyData1.to_sql("SurveyData1", con, if_exists='replace')

### Now let's take a survey taker who mentioned commutezip is 90210 and 30 min preferred commute

### Let's pull the survey takers record

surveytaker90210= pd.read_sql("select * from SurveyData1 where Address_Work_Zipcode= 90210 and Response=438", con)

## Clearly for this survey takers max house price is 820000 and min price is 660000

## Let's take 8% of the 820000 for the right house
a= (820000 +(820000)*.08)

print(a)

surveytaker90210['Max_PriceWillingtoPay'] = (820000 + np.std(surveytaker90210['House_Max_Price']))

### let

### Create the database
con = sqlite3.connect("MLSData.db")


MLSData.to_sql("MLSData", con, if_exists='replace')

MLSData.info()

### Extract all the Zip code from the MLS Data

MLSZIp= pd.read_sql("SELECT Houseid, Price, AddressCity, AddressState, AddressZip FROM MLSData",con)

zip90210all= pd.read_sql("select * from MLSData where AddressZip= 90210", con)

### main result
zip90210Pri= pd.read_sql("select Houseid, Price, AddressCity, AddressState, AddressZip from MLSData where ((AddressZip= 90210) and (Price>= 660000) and (Price<= 885600))", con)

### Secondary results

zip90210Sec= pd.read_sql("select Houseid, Price, AddressCity, AddressState, AddressZip from MLSData where ((AddressZip= 90210) and (Price>= 500000) and (Price<= 1100000))", con)

### Here is an approach 2

## Build a classification model using AUTOML for the Sold vs. Active on MLS

### Based on the key factors create a score then when a user sends chat we can find the zip present the houses (top 3) with the score


### Let's build a AutomL

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans 

SurveyData.info()

SurveyDataR= SurveyData[['Age', 'Rooms_Min','Rooms_Max', 'House_Min_Price','House_Max_Price' ]]

# statistics of the data
SurveyDataR.describe()

SurveyDataR.isnull().any()

SurveyDataRR = SurveyDataR.fillna(method='ffill')

SurveyDataRR =SurveyDataRR[~SurveyDataRR.isin([np.nan, np.inf, -np.inf]).any(1)]

SurveyDataRR.isnull().any()

# standardizing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaledRR = scaler.fit_transform(SurveyDataRR)

#data_scaledRR =data_scaledRR[~data_scaledRR.isin([np.nan, np.inf, -np.inf]).any(1)]

# statistics of scaled data
pd.DataFrame(data_scaledRR).describe()

# defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=2, init='k-means++')

# fitting the k means algorithm on scaled data
kmeans.fit(data_scaledRR)

# inertia on the fitted data
kmeans.inertia_

# fitting multiple k-means algorithms and storing the values in an empty list
SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaledRR)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

# k means using 5 clusters and k-means++ initialization
kmeans = KMeans(n_jobs = -1, n_clusters = 8, init='k-means++')
kmeans.fit(data_scaledRR)
pred = kmeans.predict(data_scaledRR)

frame = pd.DataFrame(data_scaledRR)
frame['cluster'] = pred
frame['cluster'].value_counts()


### Build an autoML classification 

### The main dataset is the MLS Data

MLSData.info()

MLDataModel= MLSData.iloc[:,[0,1,2,7, 13,18,19,20,22,23,28]]

MLDataModel.info()

### Sale Status has 4 values nan, Active, Sold and Pre-On_market

MLDataModel = MLDataModel[MLDataModel['Sale_Status'].notna()]

MLDataModel["Sale_Status"].value_counts()

## Filter out the Pre On-Market to make it a binary classification data

MLDataModel1 = MLDataModel.drop(MLDataModel.index[MLDataModel.Sale_Status == 'Pre On-Market'])

MLDataModel1["Sale_Status"] = MLDataModel1["Sale_Status"].astype('category')

MLDataModel1.dtypes

MLDataModel1.info()


from sklearn.model_selection import train_test_split

train2, test2 = train_test_split(MLDataModel1, test_size=0.15)


test2.info()

train_h2o = h2o.H2OFrame(train2)

#valid_h2o = h2o.H2OFrame(validate)

test_h2o = h2o.H2OFrame(test2)

#passId = test_h2o['Houseid']

#train_h2o = train_h2o.drop('passId',axis =1)
#test_h2o = test_h2o.drop('passId',axis =1)

x = train_h2o.columns
y = "Sale_Status"
x.remove(y)
x.remove('Houseid')
x.remove('AddressZip')

train_h2o['Sale_Status'] = train_h2o['Sale_Status'].asfactor()

test_h2o['Sale_Status'] = test_h2o['Sale_Status'].asfactor()

aml = H2OAutoML(max_models=12, seed=1)
aml.train(x=x, y=y, training_frame=train_h2o)

lb = h2o.automl.get_leaderboard(aml, extra_columns = 'ALL')

lb


lb1 = lb.as_data_frame()



# Get model ids for all models in the AutoML Leaderboard
model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_AllModels" in mid][0])
metalearner = h2o.get_model(se.metalearner()['name'])

metalearner.std_coef_plot()

metalearner.varimp() 

model = h2o.get_model('GBM_grid__1_AutoML_20201221_142644_model_2') 

a=model.varimp(use_pandas=True)

MLDataModel1 = MLDataModel1.to_csv (r'C:\Users\Dans-IQS\Documents\BackupfromLenovo10142020\Documents\BackupPreviousWindows\IntelliMatch\MLDataModel1.csv', index = None, header=True)

a = a.to_csv (r'C:\Users\Dans-IQS\Documents\BackupfromLenovo10142020\Documents\BackupPreviousWindows\IntelliMatch\a.csv', index = None, header=True)

### Now we can use the GBM percentage or scaled percentage to to create a score

model.model_performance(test_h2o)

pred = model.predict(test_h2o)

pred = aml.leader.predict(test_h2o)

pred_df = pred.as_data_frame()

pred_res = pred_df.predict

preds = aml.leader.predict(test_h2o)

model.varimp_plot(num_of_features = 9)

pred.head()

print(pd.__version__)
# Identify predictors and response


### MLDataModel1["Sale_Status"] = MLDataModel1["Sale_Status"].asfactor()


h2o.cluster().shutdown()

FinalMLSDataWithScore = pd.read_csv('C:/Users/Dans-IQS/Documents/BackupfromLenovo10142020/Documents/BackupPreviousWindows/IntelliMatch/FinalMLSDataWithScore.csv',encoding= 'iso-8859-1')





