import tensorflow as ts
import pandas as pd
import os
import seaborn as sns
import scipy.stats as stats
import math
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
import camelot
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
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from operator import attrgetter
from sklearn.metrics import silhouette_samples
import featuretools as ft
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
ps = lambda q: sqldf(q, globals())
from scipy.stats import pearsonr
sns.set(style='white', font_scale=1.2)

sess = ts.Session()
a = ts.constant(10)
b = ts.constant(32)
print(sess.run(a+b))


a = ts.constant(10)
b = ts.constant(32)
print(sess.run(a+b))

pysqldf = lambda q: sqldf(q, globals())


### 2020 Data starts here

DataCumRanks2020 = pd.read_csv('C:/Users/test/Documents/AnnuitySubmit3year_Refresh2020/2020Data/2020DataCumRanks.csv',encoding= 'iso-8859-1')

DataCumRanks2020.columns = DataCumRanks2020.columns.str.replace(' ', '')

DataCumRanks2020.columns = DataCumRanks2020.columns.str.lstrip()

DataCumRanks2020.columns = DataCumRanks2020.columns.str.rstrip()

DataCumRanks2020.columns = DataCumRanks2020.columns.str.strip()

con = sqlite3.connect("DataCumRanks2020.db")

DataCumRanks2020.to_sql("DataCumRanks2020", con, if_exists='replace')

DataCumRanks2020.info()

### 2019 Data starts here

### 2019 Committed

DataCommitted2019 = pd.read_csv('C:/Users/test/Documents/AnnuitySubmit3year_Refresh2020/2019Data/2019Committed.csv',encoding= 'iso-8859-1')

DataCommitted2019.columns = DataCommitted2019.columns.str.replace(' ', '')

DataCommitted2019.columns = DataCommitted2019.columns.str.lstrip()

DataCommitted2019.columns = DataCommitted2019.columns.str.rstrip()

DataCommitted2019.columns = DataCommitted2019.columns.str.strip()

con = sqlite3.connect("DataCommitted2019.db")

DataCommitted2019.to_sql("DataCommitted2019", con, if_exists='replace')

DataCommitted2019.info()

qqq122  = """SELECT a.* FROM DataCumRanks2020 a INNER JOIN DataCommitted2019 b on a.AdvisorContactIDText = b.AdvisorContactIDText;"""

Committed2019DidnotFell=  pysqldf(qqq122) 

p23  = """SELECT * FROM DataCommitted2019 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Committed2019DidnotFell);"""
        
Committed2019Fellof2020 =  pysqldf(p23)

### 3 advisors Fell of from committed

### 2019 Latent Committed

DataLatentCommitted2019 = pd.read_csv('C:/Users/test/Documents/AnnuitySubmit3year_Refresh2020/2019Data/2019LatentCommitted.csv',encoding= 'iso-8859-1')

DataLatentCommitted2019.columns = DataLatentCommitted2019.columns.str.replace(' ', '')

DataLatentCommitted2019.columns = DataLatentCommitted2019.columns.str.lstrip()

DataLatentCommitted2019.columns = DataLatentCommitted2019.columns.str.rstrip()

DataLatentCommitted2019.columns = DataLatentCommitted2019.columns.str.strip()

con = sqlite3.connect("DataLatentCommitted2019.db")

DataLatentCommitted2019.to_sql("DataLatentCommitted2019", con, if_exists='replace')

DataLatentCommitted2019.info()

qqq122  = """SELECT a.* FROM DataCumRanks2020 a INNER JOIN DataLatentCommitted2019 b on a.AdvisorContactIDText = b.AdvisorContactIDText;"""

LatentCommitted2019DidnotFell=  pysqldf(qqq122) 

p23  = """SELECT * FROM DataLatentCommitted2019 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM LatentCommitted2019DidnotFell);"""
        
LatentCommitted2019Fellof2020 =  pysqldf(p23)

### 31 advisors Fell of from Latent Committed

### 2019Casual Producers

CasualProducers2019 = pd.read_csv('C:/Users/test/Documents/AnnuitySubmit3year_Refresh2020/2019Data/2019CasualProducers.csv',encoding= 'iso-8859-1')

CasualProducers2019.columns = CasualProducers2019.columns.str.replace(' ', '')

CasualProducers2019.columns = CasualProducers2019.columns.str.lstrip()

CasualProducers2019.columns = CasualProducers2019.columns.str.rstrip()

CasualProducers2019.columns = CasualProducers2019.columns.str.strip()

con = sqlite3.connect("CasualProducers2019.db")

CasualProducers2019.to_sql("CasualProducers2019", con, if_exists='replace')

CasualProducers2019.info()

qqq122  = """SELECT a.* FROM DataCumRanks2020 a INNER JOIN CasualProducers2019 b on a.AdvisorContactIDText = b.AdvisorContactIDText;"""

CasualProducers2019DidnotFell=  pysqldf(qqq122) 

p23  = """SELECT * FROM CasualProducers2019 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM CasualProducers2019DidnotFell);"""
        
CasualProducers2019Fellof2020 =  pysqldf(p23)

### 230 advisors Fell of from Casual group

### 2019Dabblers Producers

Dabbler2019 = pd.read_csv('C:/Users/test/Documents/AnnuitySubmit3year_Refresh2020/2019Data/2019Dabblers.csv',encoding= 'iso-8859-1')

Dabbler2019.columns = Dabbler2019.columns.str.replace(' ', '')

Dabbler2019.columns = Dabbler2019.columns.str.lstrip()

Dabbler2019.columns = Dabbler2019.columns.str.rstrip()

Dabbler2019.columns = Dabbler2019.columns.str.strip()

con = sqlite3.connect("Dabbler2019.db")

Dabbler2019.to_sql("Dabbler2019", con, if_exists='replace')

Dabbler2019.info()

qqq122  = """SELECT a.* FROM DataCumRanks2020 a INNER JOIN Dabbler2019 b on a.AdvisorContactIDText = b.AdvisorContactIDText;"""

Dabbler20192019DidnotFell=  pysqldf(qqq122) 

p23  = """SELECT * FROM Dabbler2019 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Dabbler20192019DidnotFell);"""
        
Dabblers2019Fellof2020 =  pysqldf(p23)

### 743 advisors Fell of from Casual

### Top 5% Percent

TopFivePercent2019 = pd.read_csv('C:/Users/test/Documents/AnnuitySubmit3year_Refresh2020/2019Data/2019TopFivePercent.csv',encoding= 'iso-8859-1')

TopFivePercent2019.columns = TopFivePercent2019.columns.str.replace(' ', '')

TopFivePercent2019.columns = TopFivePercent2019.columns.str.lstrip()

TopFivePercent2019.columns = TopFivePercent2019.columns.str.rstrip()

TopFivePercent2019.columns = TopFivePercent2019.columns.str.strip()

con = sqlite3.connect("TopFivePercent2019.db")

TopFivePercent2019.to_sql("TopFivePercent2019", con, if_exists='replace')

TopFivePercent2019.info()

qqq122  = """SELECT a.* FROM DataCumRanks2020 a INNER JOIN TopFivePercent2019 b on a.AdvisorContactIDText = b.AdvisorContactIDText;"""

TopFivePercent2019DidnotFell=  pysqldf(qqq122) 

p23  = """SELECT * FROM TopFivePercent2019 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM TopFivePercent2019DidnotFell);"""
        
TopFivePercent2019Fellof2020 =  pysqldf(p23)

## O advisor Fell Off


### Top 6-10%% Percent

Six_TenPercent2019 = pd.read_csv('C:/Users/test/Documents/AnnuitySubmit3year_Refresh2020/2019Data/2019Six_TenPercent.csv',encoding= 'iso-8859-1')

Six_TenPercent2019.columns = Six_TenPercent2019.columns.str.replace(' ', '')

Six_TenPercent2019.columns = Six_TenPercent2019.columns.str.lstrip()

Six_TenPercent2019.columns = Six_TenPercent2019.columns.str.rstrip()

Six_TenPercent2019.columns = Six_TenPercent2019.columns.str.strip()

con = sqlite3.connect("Six_TenPercent2019.db")

Six_TenPercent2019.to_sql("Six_TenPercent2019", con, if_exists='replace')

Six_TenPercent2019.info()

qqq122  = """SELECT a.* FROM DataCumRanks2020 a INNER JOIN Six_TenPercent2019 b on a.AdvisorContactIDText = b.AdvisorContactIDText;"""

Six_TenPercent2019DidnotFell=  pysqldf(qqq122) 

con = sqlite3.connect("Six_TenPercent2019DidnotFell.db")

Six_TenPercent2019.to_sql("Six_TenPercent2019DidnotFell", con, if_exists='replace')

p23  = """SELECT * FROM Six_TenPercent2019 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Six_TenPercent2019DidnotFell);"""
        
Six_TenPercent2019Fellof2020=  pysqldf(p23)

## 1 advisor Fell Off

### Ten_Twenty Percent


Ten_TwentyPercent2019 = pd.read_csv('C:/Users/test/Documents/AnnuitySubmit3year_Refresh2020/2019Data/2019Ten_TwentyPercent.csv',encoding= 'iso-8859-1')

Ten_TwentyPercent2019.columns = Ten_TwentyPercent2019.columns.str.replace(' ', '')

Ten_TwentyPercent2019.columns = Ten_TwentyPercent2019.columns.str.lstrip()

Ten_TwentyPercent2019.columns = Ten_TwentyPercent2019.columns.str.rstrip()

Ten_TwentyPercent2019.columns = Ten_TwentyPercent2019.columns.str.strip()

con = sqlite3.connect("Ten_TwentyPercent2019.db")

Ten_TwentyPercent2019.to_sql("Ten_TwentyPercent2019", con, if_exists='replace')

Ten_TwentyPercent2019.info()

qqq122  = """SELECT a.* FROM DataCumRanks2020 a INNER JOIN Ten_TwentyPercent2019 b on a.AdvisorContactIDText = b.AdvisorContactIDText;"""

Ten_TwentyPercent2019DidnotFell=  pysqldf(qqq122) 

con = sqlite3.connect("Ten_TwentyPercent2019DidnotFell.db")

Ten_TwentyPercent2019DidnotFell.to_sql("Ten_TwentyPercent2019DidnotFell", con, if_exists='replace')

p23  = """SELECT * FROM Ten_TwentyPercent2019 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Ten_TwentyPercent2019DidnotFell);"""
        
Ten_TwentyPercent2019Fellof2020=  pysqldf(p23)

## 2 advisor Fell Off

### Producers $5M and more

Producer5MMore = pd.read_csv('C:/Users/test/Documents/AnnuitySubmit3year_Refresh2020/2019Data/Producer5BMore.csv',encoding= 'iso-8859-1')

Producer5MMore.columns = Producer5MMore.columns.str.replace(' ', '')

Producer5MMore.columns = Producer5MMore.columns.str.lstrip()

Producer5MMore.columns = Producer5MMore.columns.str.rstrip()

Producer5MMore.columns = Producer5MMore.columns.str.strip()

con = sqlite3.connect("Producer5MMore.db")

Producer5MMore.to_sql("Producer5MMore", con, if_exists='replace')

Producer5MMore.info()

qqq122  = """SELECT a.* FROM DataCumRanks2020 a INNER JOIN Producer5MMore b on a.AdvisorContactIDText = b.AdvisorContactIDText;"""

Producer5MMore2019DidnotFell=  pysqldf(qqq122) 

con = sqlite3.connect("Producer5MMore2019DidnotFell.db")

Producer5MMore2019DidnotFell.to_sql("Producer5MMore2019DidnotFell", con, if_exists='replace')

p23  = """SELECT * FROM Producer5MMore WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Producer5MMore2019DidnotFell);"""
        
Producer5MMore2019Fellof2020=  pysqldf(p23)


### Producers $3M and more

Producer3MMore = pd.read_csv('C:/Users/test/Documents/AnnuitySubmit3year_Refresh2020/2019Data/Producer3MMore.csv',encoding= 'iso-8859-1')

Producer3MMore.columns = Producer3MMore.columns.str.replace(' ', '')

Producer3MMore.columns = Producer3MMore.columns.str.lstrip()

Producer3MMore.columns = Producer3MMore.columns.str.rstrip()

Producer3MMore.columns = Producer3MMore.columns.str.strip()

con = sqlite3.connect("Producer3MMore.db")

Producer3MMore.to_sql("Producer3MMore", con, if_exists='replace')

Producer3MMore.info()

qqq122  = """SELECT a.* FROM DataCumRanks2020 a INNER JOIN Producer3MMore b on a.AdvisorContactIDText = b.AdvisorContactIDText;"""

Producer3MMore2019DidnotFell=  pysqldf(qqq122) 

con = sqlite3.connect("Producer3MMore2019DidnotFell.db")

Producer3MMore2019DidnotFell.to_sql("Producer3MMore2019DidnotFell", con, if_exists='replace')

p23  = """SELECT * FROM Producer3MMore WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Producer3MMore2019DidnotFell);"""
        
Producer3MMore2019Fellof2020=  pysqldf(p23)

### Let's look into the lost opportunity 

TenSeventyPercent = pd.read_csv('C:/Users/test/Documents/AnnuitySubmit3year_Refresh2020/2019Data/TenSeventyPercent.csv',encoding= 'iso-8859-1')

TenSeventyPercent.columns = TenSeventyPercent.columns.str.replace(' ', '')

TenSeventyPercent.columns = TenSeventyPercent.columns.str.lstrip()

TenSeventyPercent.columns = TenSeventyPercent.columns.str.rstrip()

TenSeventyPercent.columns = TenSeventyPercent.columns.str.strip()

con = sqlite3.connect("TenSeventyPercent.db")

TenSeventyPercent.to_sql("TenSeventyPercent", con, if_exists='replace')

TenSeventyPercent.info()

qqq122  = """SELECT a.* FROM DataCumRanks2020 a INNER JOIN TenSeventyPercent b on a.AdvisorContactIDText = b.AdvisorContactIDText;"""

TenSeventyPercent2019DidnotFell=  pysqldf(qqq122) 

con = sqlite3.connect("TenSeventyPercent2019DidnotFell.db")

TenSeventyPercent2019DidnotFell.to_sql("TenSeventyPercent2019DidnotFell", con, if_exists='replace')

p23  = """SELECT * FROM TenSeventyPercent WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM TenSeventyPercent2019DidnotFell);"""
        
ProducerTenSeventyPercent2019Fellof2020=  pysqldf(p23)


Out5 = ProducerTenSeventyPercent2019Fellof2020.to_csv(r'C:\Users\test\Documents\AnnuitySubmit3year_Refresh2020\2019Data\ProducerTenSeventyPercent2019Fellof2020.csv', index = None, header=True)


### Let's look into the drop off between 2018 and 2019

## 2019 data

Allproducers2019 = pd.read_csv('C:/Users/test/Documents/AnnuitySubmit3year_Refresh2020/2019Data/Allproducers2019.csv',encoding= 'iso-8859-1')

Allproducers2019.columns = Allproducers2019.columns.str.replace(' ', '')

Allproducers2019.columns = Allproducers2019.columns.str.lstrip()

Allproducers2019.columns = Allproducers2019.columns.str.rstrip()

Allproducers2019.columns = Allproducers2019.columns.str.strip()

con = sqlite3.connect("Allproducers2019.db")

Allproducers2019.to_sql("Allproducers2019", con, if_exists='replace')

Allproducers2019.info()

### 2018 Data starts here

### 2018 Committed

Committed2018 = pd.read_csv('C:/Users/test/Documents/AnnuitySubmit3year_Refresh2020/2018Data/Committed2018.csv',encoding= 'iso-8859-1')

Committed2018.columns = Committed2018.columns.str.replace(' ', '')

Committed2018.columns = Committed2018.columns.str.lstrip()

Committed2018.columns = Committed2018.columns.str.rstrip()

Committed2018.columns = Committed2018.columns.str.strip()

con = sqlite3.connect("Committed2018.db")

Committed2018.to_sql("Committed2018", con, if_exists='replace')

Committed2018.info()

qqq122  = """SELECT a.* FROM Allproducers2019 a INNER JOIN Committed2018 b on a.AdvisorContactIDText = b.AdvisorContactIDText;"""

Committed2018DidnotFell=  pysqldf(qqq122) 

p23  = """SELECT * FROM Committed2018 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Committed2018DidnotFell);"""
        
Committed2018Fellof2019 =  pysqldf(p23)

### 1 Advisor Fell off

### 2018 Latent Committed

LatentCommitted2018 = pd.read_csv('C:/Users/test/Documents/AnnuitySubmit3year_Refresh2020/2018Data/LatentCommitted2018.csv',encoding= 'iso-8859-1')

LatentCommitted2018.columns = LatentCommitted2018.columns.str.replace(' ', '')

LatentCommitted2018.columns = LatentCommitted2018.columns.str.lstrip()

LatentCommitted2018.columns = LatentCommitted2018.columns.str.rstrip()

LatentCommitted2018.columns = LatentCommitted2018.columns.str.strip()

con = sqlite3.connect("LatentCommitted2018.db")

LatentCommitted2018.to_sql("LatentCommitted2018", con, if_exists='replace')

LatentCommitted2018.info()

qqq122  = """SELECT a.* FROM Allproducers2019 a INNER JOIN LatentCommitted2018 b on a.AdvisorContactIDText = b.AdvisorContactIDText;"""

LatentCommitted2018DidnotFell=  pysqldf(qqq122) 

p23  = """SELECT * FROM LatentCommitted2018 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM LatentCommitted2018DidnotFell);"""
        
LatentCommitted2018Fellof2019 =  pysqldf(p23)

### 7 Advisor Fell off

### 2018 Casual Producers

Casual2018 = pd.read_csv('C:/Users/test/Documents/AnnuitySubmit3year_Refresh2020/2018Data/Casual2018.csv',encoding= 'iso-8859-1')

Casual2018.columns = Casual2018.columns.str.replace(' ', '')

Casual2018.columns = Casual2018.columns.str.lstrip()

Casual2018.columns = Casual2018.columns.str.rstrip()

Casual2018.columns = Casual2018.columns.str.strip()

con = sqlite3.connect("Casual2018.db")

Casual2018.to_sql("Casual2018", con, if_exists='replace')

Casual2018.info()

qqq122  = """SELECT a.* FROM Allproducers2019 a INNER JOIN Casual2018 b on a.AdvisorContactIDText = b.AdvisorContactIDText;"""

Casual2018DidnotFell=  pysqldf(qqq122) 

p23  = """SELECT * FROM Casual2018 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Casual2018DidnotFell);"""
        
Casual2018Fellof2019 =  pysqldf(p23)

### 84 Advisor Fell off

### 2018 Dabblers

Dabbler2018 = pd.read_csv('C:/Users/test/Documents/AnnuitySubmit3year_Refresh2020/2018Data/Dabbler2018.csv',encoding= 'iso-8859-1')

Dabbler2018.columns = Dabbler2018.columns.str.replace(' ', '')

Dabbler2018.columns = Dabbler2018.columns.str.lstrip()

Dabbler2018.columns = Dabbler2018.columns.str.rstrip()

Dabbler2018.columns = Dabbler2018.columns.str.strip()

con = sqlite3.connect("Dabbler2018.db")

Dabbler2018.to_sql("Dabbler2018", con, if_exists='replace')

Dabbler2018.info()

qqq122  = """SELECT a.* FROM Allproducers2019 a INNER JOIN Dabbler2018 b on a.AdvisorContactIDText = b.AdvisorContactIDText;"""

Dabbler2018DidnotFell=  pysqldf(qqq122) 

p23  = """SELECT * FROM Dabbler2018 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Dabbler2018DidnotFell);"""
        
Dabbler2018Fellof2019 =  pysqldf(p23)

### 1957 Advisor Fell off

### Let's take 2019 Dabblers i.e., Dabbler2019

X = Dabbler2019.iloc[:, [1,2]].values #colonne che mi interessano

#Find the number of clusters
import matplotlib.pyplot as plot
wcss = []

for i in range (1,16): #15 cluster
    kmeans = KMeans(n_clusters = i, init='k-means++', random_state=0) 
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plot.plot(range(1,16),wcss)
plot.title('Elbow Method')
plot.xlabel('Number of clusters')
plot.ylabel('wcss')
plot.show()

#KMeans clustering
kmeans= KMeans(n_clusters=5,init='k-means++', random_state=0)
y=kmeans.fit_predict(X)

plot.scatter(X[y == 0,0], X[y==0,1], s=25, c='red', label='Cluster 1')
plot.scatter(X[y == 1,0], X[y==1,1], s=25, c='blue', label='Cluster 2')
plot.scatter(X[y == 2,0], X[y==2,1], s=25, c='magenta', label='Cluster 3')
plot.scatter(X[y == 3,0], X[y==3,1], s=25, c='cyan', label='Cluster 4')
plot.scatter(X[y == 4,0], X[y==4,1], s=25, c='green', label='Cluster 5')

plot.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=25, c='yellow', label='Centroid')
plot.title('Clusters Within Dabblers')
plot.xlabel('Submitted Business')
plot.ylabel('Number of Tickets')
plot.legend()
plot.show()

# array of indexes corresponding to classes around centroids, in the order of your dataset
classified_data = kmeans.labels_

#copy dataframe (may be memory intensive but just for illustration)
df_processed = Dabbler2019.copy()
df_processed['Cluster Class'] = pd.Series(classified_data, index=df_processed.index)

Out5 = df_processed.to_csv(r'C:\Users\test\Documents\AnnuitySubmit3year_Refresh2020\2019Data\df_processed.csv', index = None, header=True)

