
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
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer   
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
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


## Dataset 1

AnnuitySub3YrsMar = pd.read_csv('C:/Users/test/Documents/AnnuitySubmits_3Years/Salesforce/Submit_08012016until08012019.csv',encoding= 'iso-8859-1')

AnnuitySub3YrsMar.columns= AnnuitySub3YrsMar.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

AnnuitySub3YrsMar.info()

### Change the date format

AnnuitySub3YrsMar['submit_date'] = pd.to_datetime(AnnuitySub3YrsMar['submit_date'])

### Create the database
con = sqlite3.connect("AnnuitySub3YrsMar.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
AnnuitySub3YrsMar.to_sql("AnnuitySub3YrsMar", con, if_exists='replace')

### Producers(Advissors) information: let's cut the data based on 3 separate years first

AnnuitySub3YrsMar = pd.read_sql("SELECT marketer_marketer_name as marketer_name, submit_date, submit_amount FROM AnnuitySub3YrsMar ",con)

exp_csv = AnnuitySub3YrsMar.to_csv (r'C:\Users\test\Documents\AnnuitySubmits_3Years\Salesforce\Marketers\AnnuitySub3YrsMar.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

### August 2016 until July 2017
Ads_Aug2016_July2017Mar = pd.read_sql("SELECT marketer_marketer_name as marketer_name, submit_date, submit_amount FROM AnnuitySub3YrsMar where submit_date>= '2016-08-01' and submit_date < '2017-08-01' order by submit_date",con)

exp1_csv = Ads_Aug2016_July2017Mar.to_csv (r'C:\Users\test\Documents\AnnuitySubmits_3Years\Salesforce\Marketers\Ads_Aug2016_July2017Mar.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

### August 2017 until July 2018

Ads_Aug2017_July2018Mar = pd.read_sql("SELECT marketer_marketer_name as marketer_name, submit_date, submit_amount FROM AnnuitySub3YrsMar where submit_date>= '2017-08-01' and submit_date < '2018-08-01' order by submit_date",con)

exp2_csv = Ads_Aug2017_July2018Mar.to_csv (r'C:\Users\test\Documents\AnnuitySubmits_3Years\Salesforce\Marketers\Ads_Aug2017_July2018Mar.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

### August 2018 until Aug 1st 2019

Ads_Aug2018_Aug1st2019Mar = pd.read_sql("SELECT marketer_marketer_name as marketer_name, submit_date, submit_amount FROM AnnuitySub3YrsMar where submit_date>= '2018-08-01' and submit_date < '2019-08-02' order by submit_date",con)

exp3_csv = Ads_Aug2018_Aug1st2019Mar.to_csv (r'C:\Users\test\Documents\AnnuitySubmits_3Years\Salesforce\Marketers\Ads_Aug2018_Aug1st2019Mar.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

### Trying out DBSCan algorithm

X=Ads_Aug2018_Aug1st2019Mar.submit_amount

X1=np.reshape(X, 40950)

X1 = X1.reshape(1, -1) 

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X1)





