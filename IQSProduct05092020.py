# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 22:09:51 2020

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
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
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

Sourcemedium_DF = pd.read_csv('C:/Users/DanSarkar/Documents/BackupPreviousWindows/BusinessDevelopment/IQSProductRoadmap/Data/SourceMediumDataFinalDetails.csv',encoding= 'iso-8859-1')

##IllusDF.columns= AnnuitySub3YrsMar.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

Sourcemedium_DF.info()

Sourcemedium_DF['PercentUsers']= ((Sourcemedium_DF['PercentUsers'])*100)

###Sourcemedium_DF[["AvgSessionDuration"]] = Sourcemedium_DF[["AvgSessionDuration"]].apply(pd.to_numeric)

Sourcemedium_DF['PagesPerSession'].describe()

Sourcemedium_DF['Transactions'].describe()

### Create the database
con = sqlite3.connect("Sourcemedium_DF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Sourcemedium_DF.to_sql("Sourcemedium_DF", con, if_exists='replace')

### Producers(Advissors) information: let's cut the data based on 3 separate years first

Sourcemedium_DFcount = pd.read_sql("SELECT Users, PercentUsers, Source_Medium FROM Sourcemedium_DF order by Users desc limit 5",con)


###Sourcemedium_DF['Users'].value_counts()[:20].plot(kind='barh')


ASAPData_Medium = pd.read_csv('C:/Users/DanSarkar/Documents/BackupPreviousWindows/BusinessDevelopment/IQSProductRoadmap/Data/ASAPData_Medium.csv',encoding= 'iso-8859-1')


### Create the database
con = sqlite3.connect("ASAPData_Medium.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
ASAPData_Medium.to_sql("ASAPData_Medium", con, if_exists='replace')


ASAPData_Medium.info()

ASAPData_Medium['Date'] = pd.to_datetime(ASAPData_Medium['Date'])

TotalEmail = pd.read_sql("SELECT sum(Email) as TotalEmail, Total FROM ASAPData_Medium group by Email",con)

sums = ASAPData_Medium.select_dtypes(pd.np.number).sum().rename('total1')

ASAPData_Medium.append(sums)

ASAPData_Medium.loc['total1'] = ASAPData_Medium.select_dtypes(pd.np.number).sum()


ASAPData_Medium['EmailProp'] = sum(ASAPData_Medium['Email'])/sum(ASAPData_Medium['Total'])


ASAPData_Medium['InstagramProp'] = sum(ASAPData_Medium['Instagram'])/sum(ASAPData_Medium['Total'])


ASAPData_Medium['YelpProp'] = sum(ASAPData_Medium['Yelp'])/sum(ASAPData_Medium['Total'])

ASAPData_Medium['EmailPercent']= int((sum(ASAPData_Medium['Email'])/sum(ASAPData_Medium['Total']))*100)

ASAPData_Medium['InstagramPercent']= int((sum(ASAPData_Medium['Instagram'])/sum(ASAPData_Medium['Total']))*100)

ASAPData_Medium['YelpPercent']= int((sum(ASAPData_Medium['Yelp'])/sum(ASAPData_Medium['Total']))*100)


data = {'Medium':['Email', 'Instagram', 'Yelp'], 'Percent Time':[76, 48, 23]} 
  
# Create DataFrame 
df = pd.DataFrame(data) 
  
# Print the output. 
print(df) 

dataFrame  = pd.DataFrame(data = df);

colors = ['green','cyan','red',]

ax= dataFrame.plot.barh(x='Medium', y='Percent Time', title="Your audience spend a percentage of time on", color=colors);

plt.savefig('ax.png')


## ax = additives_count.plot(x='Medium', y='Percent Time',kind='barh',color=dataFrame['Colors'])

ax.show(block=True);




