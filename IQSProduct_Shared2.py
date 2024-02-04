# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 19:36:39 2020

@author: DanSarkar

This shows the protype of how to match Intellimatch Survey data with MLS data based on multiple calculations.

The prototype is defined from multiiple models and calculations and it is a proof of concept that has been valiadated
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
h2o.init(max_mem_size='2G')
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

InitialToolData = pd.read_csv('C:/Users/Dans-IQS/Documents/BackupfromLenovo10142020/BusinessDevelopment/IQSProductRoadmap/Data/newCampaignUpdated.csv',encoding= 'iso-8859-1')

InitialToolData2 = pd.read_csv('C:/Users/Dans-IQS/Documents/BackupfromLenovo10142020/BusinessDevelopment/IQSProductRoadmap/Data/newCampaignUpdated2.csv',encoding= 'iso-8859-1')

InitialToolData2.info()


### Build an autoML classification 


InitialToolData3= InitialToolData2.iloc[:,[2,3,4,8]]

InitialToolData3.info()

InitialToolData3["productService"] = InitialToolData3["productService"].astype('category')

InitialToolData3["primaryGoal"] = InitialToolData3["primaryGoal"].astype('category')


from sklearn.model_selection import train_test_split

train2, test2 = train_test_split(InitialToolData3, test_size=0.25)


test2.info()

train_h2o = h2o.H2OFrame(train2)



test_h2o = h2o.H2OFrame(test2)


x = train_h2o.columns
y = "numberofCampaign"
x.remove(y)

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

### This part of the code will change so this needs to be updated as needed
# Get model ids for all models in the AutoML Leaderboard

model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])

# Get the GBM model
model2 = h2o.get_model([mid for mid in model_ids if "GBM" in mid][0])  

### This is the coffeficent that 


a2=model2.varimp(use_pandas=True)
a3=model2.varimp()

pred = model2.predict(test_h2o)

pred1 = pred.as_data_frame()

Pred2= round(pred1)

test2_copy= test2.copy()

test2['PredictedNumberofCampaign']= pred1['predict']

test2['PredictedNumberofCampaignround']= round(test2['PredictedNumberofCampaign'])

test2.reset_index(inplace=True)

export_csv = test2.to_csv(r'C:\Users\Dans-IQS\Documents\BackupfromLenovo10142020\BusinessDevelopment\IQSProductRoadmap\Data\test2.csv', index = None, header=True)

import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)

import plotly.express as px
from plotly.offline import plot 
iris = px.data.iris()
scatter_plot = px.bar(a2, x="variable", y="percentage")
plot(scatter_plot)


#### 
InitialToolData4 = pd.read_csv('C:/Users/Dans-IQS/Documents/BackupfromLenovo10142020/BusinessDevelopment/IQSProductRoadmap/Data/newCampaignUpdated4.csv',encoding= 'iso-8859-1')

InitialToolData4.info()

InitialToolData4= InitialToolData4.iloc[:,[2,3,4,18]]

InitialToolData4.info()

InitialToolData4["productService"] = InitialToolData4["productService"].astype('category')

InitialToolData4["primaryGoal"] = InitialToolData4["primaryGoal"].astype('category')


from sklearn.model_selection import train_test_split

train21, test21 = train_test_split(InitialToolData4, test_size=0.25)


test21.info()

train_h2o1 = h2o.H2OFrame(train21)



test_h2o1 = h2o.H2OFrame(test21)


x1 = train_h2o1.columns
y1 = "AvgCampaignWeekRange"
x1.remove(y1)

aml1 = H2OAutoML(max_models=12, seed=1)
aml1.train(x=x1, y=y1, training_frame=train_h2o1)

lb21 = h2o.automl.get_leaderboard(aml1, extra_columns = 'ALL')

lb21

lb12 = lb21.as_data_frame()

# Get model ids for all models in the AutoML Leaderboard
model_ids2 = list(aml1.leaderboard['model_id'].as_data_frame().iloc[:,0])
se = h2o.get_model([mid for mid in model_ids2 if "StackedEnsemble_AllModels" in mid][0])
metalearner2 = h2o.get_model(se.metalearner()['name'])

metalearner2.std_coef_plot()

metalearner2.varimp() 

### This part of the code will change so this needs to be updated as needed
# Get model ids for all models in the AutoML Leaderboard

model_ids2 = list(aml1.leaderboard['model_id'].as_data_frame().iloc[:,0])

# Get the GBM model
model222 = h2o.get_model([mid for mid in model_ids2 if "GBM" in mid][0])  

### This is the coffeficent that 


a2222=model222.varimp(use_pandas=True)
a3333=model222.varimp()

pred_a = model222.predict(test_h2o1)

pred1_a = pred_a.as_data_frame()

Pred2_a= round(pred1_a)

test2_copy1= test21.copy()

test21['PredictedAvgCampaignWeekRange']= Pred2_a['predict']

test21['PredictedAvgCampaignWeekRange']= round(test21['PredictedAvgCampaignWeekRange'])

test2.reset_index(inplace=True)


h2o.cluster().shutdown()


