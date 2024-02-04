
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

### Cancelation List from Athene

Data_DF = pd.read_csv(r'C:\Users\test\Documents\Schillar\AnnexusLOPMailing.csv',encoding= 'iso-8859-1')

### Fallen Angel list from Dan Previous
Data_DF1 = pd.read_csv(r'C:\Users\test\Documents\Schillar\DansList.csv',encoding= 'iso-8859-1')

Data_DF['AgentNameSpace'] = Data_DF['AgentName'].str.replace(" ","")

Data_DF1['AdvisorNameSpace'] = Data_DF1['AdvisorName'].str.replace(" ","")

## Data_DF['AgentName_AtheneSharedSpace'] = Data_DF['AgentName_AtheneShared'].str.replace(" ","")

Data_DF.info()

con = sqlite3.connect("Data_DF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Data_DF.to_sql("Data_DF", con, if_exists='replace')

Data_DF.info()

con = sqlite3.connect("Data_DF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Data_DF1.to_sql("Data_DF1", con, if_exists='replace')

Data_DF1.info()

q2  = """SELECT * FROM Data_DF1 c
        INNER JOIN Data_DF d on (d.AgentName = c.AdvisorName);"""
        
Match =  pysqldf(q2) 

export_csv = Match.to_csv (r'C:\Users\test\Documents\Schillar\Match.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

### Extract the dataset which are not a match in the original dataset

Check1 = Data_DF1[~Data_DF1['AdvisorName'].isin(Match['AdvisorName'])]

export_csv = Check1.to_csv (r'C:\Users\test\Documents\Schillar\Check1.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

#### Try to do the extract with a SQL using NOT IN

 ### First create a database of the match 
 
con = sqlite3.connect("Match.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Data_DF1.to_sql("Match", con, if_exists='replace')

w2  = """SELECT * FROM Data_DF1 WHERE AdvisorName NOT IN (SELECT AdvisorName FROM Match);"""
        
Remaining =  pysqldf(w2)        

export_csv = Remaining.to_csv (r'C:\Users\test\Documents\Schillar\Remaining.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

 
####  Shiller Non-Signed Booked

Data_DF2 = pd.read_csv(r'C:\Users\test\Documents\Schillar\ShillerRemainingNon_Signed.csv',encoding= 'iso-8859-1')

Data_DF2.info()
con = sqlite3.connect("Data_DF2.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Data_DF2.to_sql("Data_DF2", con, if_exists='replace')

q3  = """SELECT * FROM Data_DF c
        INNER JOIN Data_DF2 d on (c.AgentName = d.AdvisorsName);"""
        
Match1 =  pysqldf(q3)  


####  Shiller Signed Booked

Data_DF3 = pd.read_csv(r'C:\Users\test\Documents\Schillar\ShillerTop200_Signed.csv',encoding= 'iso-8859-1')

Data_DF3.info()
con = sqlite3.connect("Data_DF3.db")

q4  = """SELECT * FROM Data_DF c
        INNER JOIN Data_DF3 d on (c.AgentName = d.AdvisorsName);"""
        
Match2 =  pysqldf(q4)  

### Let's bring the Submit data

Data_Sub1 = pd.read_csv(r'C:\Users\test\Documents\Schillar\MissingSub3monthClean2.csv',encoding= 'iso-8859-1')

Data_Sub1.info()
con = sqlite3.connect("Data_Sub1.db")

# Let's bring the Submit data Clean2

Data_Sub1.to_sql("Data_Sub1", con, if_exists='replace')

q5  = """SELECT * FROM Data_Sub1 c
        INNER JOIN Data_DF1 d on (c.Athene_AdvisorNameNotExistLast3Months = d.AdvisorName);"""
        
MatchAgent =  pysqldf(q5)

## Data_Sub = pd.read_csv(r'C:\Users\test\Documents\Schillar\MissingAtheneAdvisorLast3monthsSubmits.csv',encoding= 'iso-8859-1')
  












