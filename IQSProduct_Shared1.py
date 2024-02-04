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
a = ts.constant(10)
b = ts.constant(32)
print((a+b))

pysqldf = lambda q: sqldf(q, globals())

SurveyData = pd.read_csv('C:/Users/Dans-IQS/Documents/BackupfromLenovo10142020/BusinessDevelopment/IQSProductRoadmap/Data/ASAPData_Medium1model.csv',encoding= 'iso-8859-1')

##SurveyData1=SurveyData[~SurveyData['Address_Work_Zipcode'].isnull()]

###MLSData = pd.read_csv('C:/Users/Dans-IQS/Documents/BackupfromLenovo10142020/Documents/BackupPreviousWindows/IntelliMatch/Data/DataFr_flat_Cleaned_modi2.csv',encoding= 'iso-8859-1')

###

SurveyData.info()



MLDataModel= SurveyData


MLDataModel.info()

MLDataModel = MLDataModel[MLDataModel['TopChannel1R'].notna()]

MLDataModel["TopChannel1R"].value_counts()


##MLDataModel1 = MLDataModel.drop(MLDataModel.index[MLDataModel.Sale_Status == 'Pre On-Market'])

MLDataModel["TopChannel1R"] = MLDataModel["TopChannel1R"].astype('category')

MLDataModel.dtypes

MLDataModel.info()

from sklearn.model_selection import train_test_split

train2, test2 = train_test_split(MLDataModel, test_size=0.15)


test2.info()

train_h2o = h2o.H2OFrame(train2)

#valid_h2o = h2o.H2OFrame(validate)

test_h2o = h2o.H2OFrame(test2)

x = train_h2o.columns
y = "TopChannel1R"
x.remove(y)
x.remove('Date')
x.remove('TopChannel1')
x.remove('TopChannel2')
x.remove('TopChannel2R')

train_h2o['TopChannel1R'] = train_h2o['TopChannel1R'].asfactor()

test_h2o['TopChannel1R'] = test_h2o['TopChannel1R'].asfactor()

aml = H2OAutoML(max_models=12, seed=1)
aml.train(x=x, y=y, training_frame=train_h2o)

lb = h2o.automl.get_leaderboard(aml, extra_columns = 'ALL')

lb

lb1 = lb.as_data_frame()


model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_AllModels" in mid][0])
metalearner = h2o.get_model(se.metalearner()['name'])

metalearner.std_coef_plot()

metalearner.varimp() 

### This part of the code will change so this needs to be updated as needed
model = h2o.get_model('GBM_grid__1_AutoML_20210405_184414_model_2') 


a1=model.varimp(use_pandas=True)

###Compute the score from the Gradient Boost Percentage

MLDataModel1.info()

MLDataModel1['Score']= MLDataModel1['Price']*0.1748 + MLDataModel1['PricePerSquareFeet']* 0.0507 + MLDataModel1['NoBeds']*0.0204 + MLDataModel1['NoBathUpdated']*0.0243 + MLDataModel1['SquareFeet']*0.0787 + MLDataModel1['LotSize']*0.0638 + MLDataModel1['Expenses_HOA']*0.0502 +MLDataModel1['Sale_DaysOnMarket']*0.5366


MLDataModel2= MLDataModel1  


con = sqlite3.connect("MLDataModel2.db")

MLDataModel2.to_sql("MLDataModel2", con, if_exists='replace')

### Pull ot the zip code 90210 with Active based on survey user preference of communte within 15 min

zip90210Secmodelscore= pd.read_sql("select Houseid, Score, Sale_Status, Price, AddressZip from MLDataModel2 where ((AddressZip= 90210) and Sale_Status='Active') order by score desc", con)

zip90210Secmodelscoretop5= zip90210Secmodelscore.head(5)


MLDataModel1 = MLDataModel1.to_csv (r'C:\Users\Dans-IQS\Documents\BackupfromLenovo10142020\Documents\BackupPreviousWindows\IntelliMatch\MLDataModel1.csv', index = None, header=True)



model.model_performance(test_h2o)

pred = model.predict(test_h2o)

pred = aml.leader.predict(test_h2o)

pred_df = pred.as_data_frame()

pred_res = pred_df.predict

preds = aml.leader.predict(test_h2o)

model.varimp_plot(num_of_features = 9)

pred.head()

print(pd.__version__)

h2o.cluster().shutdown()

FinalMLSDataWithScore = pd.read_csv('C:/Users/Dans-IQS/Documents/BackupfromLenovo10142020/Documents/BackupPreviousWindows/IntelliMatch/FinalMLSDataWithScore.csv',encoding= 'iso-8859-1')

### 3


SurveyDataRR1= SurveyDataRR

kmeans = KMeans(n_jobs = -1, n_clusters = 8, init='k-means++')
kmeans.fit(SurveyDataRR1)
pred = kmeans.predict(SurveyDataRR1)

frame1 = pd.DataFrame(SurveyDataRR1)
frame1['cluster'] = pred
frame1['cluster'].value_counts()

SurveyDataR1= SurveyData[['Response', 'Address_Work_Zipcode','Age', 'Rooms_Min','Rooms_Max', 'House_Min_Price','House_Max_Price']]


con = sqlite3.connect("SurveyDataR1.db")

SurveyDataR1.to_sql("SurveyDataR1", con, if_exists='replace')

con = sqlite3.connect("frame1.db")

frame1.to_sql("frame1", con, if_exists='replace')

SurveyDataR1.info()

frame1.info()

qqq  = """SELECT a.Response, a.Address_Work_Zipcode, a.House_Min_Price, a.House_Max_Price, b.cluster FROM SurveyDataR1 a
        INNER JOIN frame1 b on (a.House_Min_Price = b.House_Min_Price) and (a.House_Max_Price = b.House_Max_Price) and (a.Age=b.Age);"""
        
frame1_SurveryDataR1 =  pysqldf(qqq) 

### Join frame1_SurveryDataR1 with MLS data based on Zip

con = sqlite3.connect("frame1_SurveryDataR1.db")

frame1_SurveryDataR1.to_sql("frame1_SurveryDataR1", con, if_exists='replace')

con = sqlite3.connect("frame1_SurveryDataR1.db")

frame1_SurveryDataR1.to_sql("frame1_SurveryDataR1", con, if_exists='replace')

MLDataModel1= MLDataModel2
frame1_SurveryDataR1.info()

con = sqlite3.connect("MLDataModel1.db")

MLDataModel1.to_sql("MLDataModel1", con, if_exists='replace')

qqq  = """SELECT  a.Houseid, a.Price, a.Score, Sale_Status, a.AddressZip, b.Response, b.Address_Work_Zipcode, b.House_Min_Price, b.House_Max_Price, b.cluster FROM 
         MLDataModel1 a LEFT JOIN frame1_SurveryDataR1 b on (a.AddressZip= b.Address_Work_Zipcode);"""
        
Cluster_ScoreCombined =  pysqldf(qqq) 


con = sqlite3.connect("Cluster_ScoreCombined.db")

Cluster_ScoreCombined.to_sql("Cluster_ScoreCombined", con, if_exists='replace')


SelectionbasedonClus= pd.read_sql("select Houseid, Score, Sale_Status, Price,House_Min_Price,House_Max_Price, AddressZip, cluster from Cluster_ScoreCombined where ((AddressZip= 90210) and (Price >=House_Min_Price-40000 and Price <= House_Max_Price+200000)and Sale_Status='Active')", con)

SelectionbasedonClustop3= SelectionbasedonClus.head(3)

#### Aggregation

zip90210Sec.info()

zip90210Secmodelscoretop5.info()

SelectionbasedonClustop3.info()

DataFrame1=zip90210Sec

DataFrame2=zip90210Secmodelscoretop5

DataFrame3=SelectionbasedonClustop3

con = sqlite3.connect("MLDataModel1.db")

DataFrame1.to_sql("DataFrame1", con, if_exists='replace')

con = sqlite3.connect("DataFrame1.db")


DataFrame2.to_sql("DataFrame2", con, if_exists='replace')

con = sqlite3.connect("DataFrame2.db")


DataFrame3.to_sql("DataFrame3", con, if_exists='replace')

con = sqlite3.connect("DataFrame3.db")

qqq  = """SELECT  distinct(a.Houseid), a.Price FROM DataFrame1 a INNER JOIN DataFrame3 c INNER Join DataFrame2 b on ((a.Houseid=c.Houseid) or (a.Houseid=c.Houseid= b.Houseid));"""
        
FinalDataFrame =  pysqldf(qqq) 
