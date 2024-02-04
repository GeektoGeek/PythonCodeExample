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

### Athene Coolers

### Load the Cooler list

### AugustNYC2019 Data

JuneNYC2019 = pd.read_csv('C:/Users/test/Documents/CventDataCameFromJoe/2019/JuneNYC2019.csv',encoding= 'iso-8859-1')

JuneNYC2019.columns = JuneNYC2019.columns.str.replace(' ', '')

### q3  = """SELECT * from AugustNYC2019 where QuestionText= 'What are the primary carriers you write with?';"""

### DFFF1_AugustNYC2019 =  pysqldf(q3) 

check = JuneNYC2019.drop_duplicates(subset = ["QuestionText"])

check= check.loc[pd.notnull(check.QuestionText)]

###Start with one question

DFFF1_JuneNYC2019U= JuneNYC2019.loc[JuneNYC2019['QuestionText'] == 'What are the primary carriers you write with?']

##filtered_df = DFFF1_AugustNYC2019U[DFFF1_AugustNYC2019U['FullName'].isnull()]

filtered_df_PC= DFFF1_JuneNYC2019U.loc[pd.notnull(DFFF1_JuneNYC2019U.FullName)]

### end of one question

###Start with question two

DFFF1_JuneNYC2019U=  JuneNYC2019.loc[JuneNYC2019['QuestionText'] == 'Please enter your Athene (Annexus) agent number.']

##filtered_df = DFFF1_AugustNYC2019U[DFFF1_AugustNYC2019U['FullName'].isnull()]

filtered_df_AgentNum= DFFF1_JuneNYC2019U.loc[pd.notnull(DFFF1_JuneNYC2019U.FullName)]

### end of question two

## merging one and two

filtered_df_combined= pd.merge(filtered_df_PC, filtered_df_AgentNum, on="EmailAddress", how='left')

filtered_df_combined.info()

### copying the question for all columns to revove the null

filtered_df_combined = filtered_df_combined.assign(QuestionText_y='Please enter your Athene (Annexus) agent number.')

## cleanup the combined filer

filtered_df_combined= filtered_df_combined.drop(['SurveyType_x'], axis = 1)

filtered_df_combined= filtered_df_combined.drop(['SurveyType_y'], axis = 1)

###Start with third question

DFFF1_JuneNYC2019U1= JuneNYC2019.loc[JuneNYC2019['QuestionText'] == 'Are you a registered rep?']

##filtered_df = DFFF1_AugustNYC2019U[DFFF1_AugustNYC2019U['FullName'].isnull()]

filtered_df_RegisteredRep= DFFF1_JuneNYC2019U1.loc[pd.notnull(DFFF1_JuneNYC2019U1.FullName)]

### end of third question

## merging three

filtered_df_combined1= pd.merge(filtered_df_combined, filtered_df_RegisteredRep, on="EmailAddress", how='left')

filtered_df_combined1= filtered_df_combined1.drop(['SurveyType'], axis = 1)

filtered_df_combined1= filtered_df_combined1.drop(['FullName_y'], axis = 1)

filtered_df_combined1= filtered_df_combined1.drop(['FullName'], axis = 1)

###Start with foruth question

DFFF1_JuneNYC2019U2= JuneNYC2019.loc[JuneNYC2019['QuestionText'] == 'Annual Annuity Premium']

##filtered_df = DFFF1_AugustNYC2019U[DFFF1_AugustNYC2019U['FullName'].isnull()]

filtered_df_AnnuityPremium= DFFF1_JuneNYC2019U2.loc[pd.notnull(DFFF1_JuneNYC2019U2.FullName)]

### end of fourth question

## merging four

filtered_df_combined2= pd.merge(filtered_df_combined1, filtered_df_AnnuityPremium, on="EmailAddress", how='left')

filtered_df_combined2= filtered_df_combined2.drop(['SurveyType'], axis = 1)

filtered_df_combined2= filtered_df_combined2.drop(['FullName'], axis = 1)

###Start with fifth question

DFFF1_JuneNYC2019U3= JuneNYC2019.loc[JuneNYC2019['QuestionText'] == 'Are you currently contracted with Athene (through Annexus)']

##filtered_df = DFFF1_AugustNYC2019U[DFFF1_AugustNYC2019U['FullName'].isnull()]

filtered_df_CurrentlyAnneContracted= DFFF1_JuneNYC2019U3.loc[pd.notnull(DFFF1_JuneNYC2019U3.FullName)]

### end of fifth question

## merging fifth

filtered_df_combined3= pd.merge(filtered_df_combined2, filtered_df_CurrentlyAnneContracted, on="EmailAddress", how='left')

filtered_df_combined3= filtered_df_combined3.drop(['SurveyType'], axis = 1)

filtered_df_combined3= filtered_df_combined3.drop(['FullName'], axis = 1)


###Start with sixth question

DFFF1_JuneNYC2019U4= JuneNYC2019.loc[JuneNYC2019['QuestionText'] == 'Assets under management']

##filtered_df = DFFF1_AugustNYC2019U[DFFF1_AugustNYC2019U['FullName'].isnull()]

filtered_df_AssetManagement= DFFF1_JuneNYC2019U4.loc[pd.notnull(DFFF1_JuneNYC2019U4.FullName)]

### end of sixth question

## merging sixth

filtered_df_combined4= pd.merge(filtered_df_combined3, filtered_df_AssetManagement, on="EmailAddress", how='left')

filtered_df_combined4= filtered_df_combined4.drop(['SurveyType'], axis = 1)

filtered_df_combined4= filtered_df_combined4.drop(['FullName'], axis = 1)

###Start with seventh question

DFFF1_JuneNYC2019U5= JuneNYC2019.loc[JuneNYC2019['QuestionText'] == 'Please select your shirt size.']

##filtered_df = DFFF1_AugustNYC2019U[DFFF1_AugustNYC2019U['FullName'].isnull()]

filtered_df_tshirtSize= DFFF1_JuneNYC2019U5.loc[pd.notnull(DFFF1_JuneNYC2019U5.FullName)]

### end of seventh question

## merging seventh

filtered_df_combined5= pd.merge(filtered_df_combined4, filtered_df_tshirtSize, on="EmailAddress", how='outer')

filtered_df_combined5= filtered_df_combined5.drop(['SurveyType'], axis = 1)

filtered_df_combined5= filtered_df_combined5.drop(['FullName'], axis = 1)

###Start with eighted question

DFFF1_JuneNYC2019U6= JuneNYC2019.loc[JuneNYC2019['QuestionText'] == 'Please enter the name of your broker dealer.']

##filtered_df = DFFF1_AugustNYC2019U[DFFF1_AugustNYC2019U['FullName'].isnull()]

filtered_df_brokerdealer= DFFF1_JuneNYC2019U6.loc[pd.notnull(DFFF1_JuneNYC2019U6.FullName)]

## merging eighted

filtered_df_combined6= pd.merge(filtered_df_combined5, filtered_df_brokerdealer, on="EmailAddress", how='outer')

filtered_df_combined6 = filtered_df_combined6.assign(QuestionText_y='Please enter the name of your broker dealer.')

filtered_df_combined6= filtered_df_combined6.drop(['SurveyType'], axis = 1)

filtered_df_combined6= filtered_df_combined6.drop(['FullName'], axis = 1)

###Start with nineth question

DFFF1_JuneNYC2019U7= JuneNYC2019.loc[JuneNYC2019['QuestionText'] == 'Contracted with Nationwide']

##filtered_df = DFFF1_AugustNYC2019U[DFFF1_AugustNYC2019U['FullName'].isnull()]

filtered_df_NWContracted= DFFF1_JuneNYC2019U7.loc[pd.notnull(DFFF1_JuneNYC2019U7.FullName)]

## merging nineth

filtered_df_combined7= pd.merge(filtered_df_combined6, filtered_df_NWContracted, on="EmailAddress", how='outer')


filtered_df_combined7= filtered_df_combined7.drop(['SurveyType'], axis = 1)

filtered_df_combined7= filtered_df_combined7.drop(['FullName'], axis = 1)

###Start with tenth question

DFFF1_JuneNYC2019U8= JuneNYC2019.loc[JuneNYC2019['QuestionText'] == 'Please enter your nw (Annexus) agent number.']

##filtered_df = DFFF1_AugustNYC2019U[DFFF1_AugustNYC2019U['FullName'].isnull()]

filtered_df_AnnexusAgent= DFFF1_JuneNYC2019U8.loc[pd.notnull(DFFF1_JuneNYC2019U8.FullName)]

## merging tenth

filtered_df_combined8= pd.merge(filtered_df_combined7, filtered_df_AnnexusAgent, on="EmailAddress", how='outer')

filtered_df_combined8 = filtered_df_combined8.assign(QuestionText_y='Please enter your nw (Annexus) agent number.')

filtered_df_combined8= filtered_df_combined8.drop(['SurveyType'], axis = 1)

filtered_df_combined8= filtered_df_combined8.drop(['FullName'], axis = 1)

filtered_df_JuneNYC2019 =filtered_df_combined8

filtered_df_JuneNYC2019["EventName"]= 'JuneNYC2019'

Out4 = filtered_df_JuneNYC2019.to_csv(r'C:/Users/test/Documents/CventDataCameFromJoe/2019/filtered_df_JuneNYC2019.csv', index = None, header=True)

filtered_df_JuneNYC2019.info()

result_check1 =pd.concat([filtered_df_JuneNYC2019,filtered_df_DecemberNYC2019], axis=0, ignore_index=True)

result_check2 =pd.concat([filtered_df_MarchNYC2019,filtered_df_AugustNYC2019], axis=0, ignore_index=True)

result_check3 =pd.concat([result_check1,result_check2], axis=0, ignore_index=True)

AllConference2019= result_check3

Out4 = AllConference2019.to_csv(r'C:/Users/test/Documents/CventDataCameFromJoe/2019/AllConference2019.csv', index = None, header=True)



