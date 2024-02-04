
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

#### Remaining States for This Year

### Let's get the data first three months of the data i.e. this years producers for non-competitive producers


Submit2019 = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/2019Submit.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Submit2019.db")

Submit2019.to_sql("Submit2019", con, if_exists='replace')

Jan1_May2_2020 = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/2020Jan1_May2.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Jan1_May2_2020.db")

Jan1_May2_2020.to_sql("Jan1_May2_2020", con, if_exists='replace')

Jan1_May2_2020.info()

q3  = """SELECT MarketerContactIDText, MarketerName, AccountName, count(SubmitDate) as SubmitCount, sum(SubmitAmount) as SubmitAmount
      FROM Jan1_May2_2020 group by MarketerContactIDText;"""
      
SubGroupByJanMay2020 =  pysqldf(q3)  

con = sqlite3.connect("SubGroupByJanMay2020.db")

SubGroupByJanMay2020.to_sql("SubGroupByJanMay2020", con, if_exists='replace')

CovidMar13_May2 = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/CovidMar13_May2.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("CovidMar13_May2.db")

CovidMar13_May2.to_sql("CovidMar13_May2", con, if_exists='replace')


q4  = """SELECT MarketerContactIDText, MarketerName, AccountName, count(SubmitDate) as SubmitCount, sum(SubmitAmount) as SubmitAmount
      FROM CovidMar13_May2 group by MarketerContactIDText;"""
      
SubGroupByCovidMar13_May2 =  pysqldf(q4) 

con = sqlite3.connect("SubGroupByCovidMar13_May2.db")

SubGroupByCovidMar13_May2.to_sql("SubGroupByCovidMar13_May2", con, if_exists='replace')

q6 = """SELECT a.MarketerContactIDText, a.MarketerName, a.AccountName FROM SubGroupByJanMay2020 a INNER JOIN SubGroupByCovidMar13_May2 b on (a.MarketerContactIDText =b.MarketerContactIDText);"""      

CommonJanndMay =  pysqldf(q6)

con = sqlite3.connect("CommonJanndMay.db")

CommonJanndMay.to_sql("CommonJanndMay", con, if_exists='replace')

q7= """SELECT * FROM SubGroupByJanMay2020 WHERE MarketerContactIDText NOT IN (SELECT MarketerContactIDText FROM CommonJanndMay);"""

MissedNames =  pysqldf(q7)

con = sqlite3.connect("MissedNames.db")

MissedNames.to_sql("MissedNames", con, if_exists='replace')

MissedNames.info()

#### Let's validate the missed names against NW Advisors appointment
### Advisors are marketers too

MarketerlistDW = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/MarketerlistDW.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("MarketerlistDW.db")

MarketerlistDW.to_sql("MarketerlistDW", con, if_exists='replace')

MarketerlistDW.info()

q88 = """SELECT a.*, b.Title, b.Status, b.DeletedFlag FROM MissedNames a INNER JOIN MarketerlistDW b on (a.MarketerContactIDText =b.SFContactId);"""      

checkval =  pysqldf(q88)

export_csv1 = checkval.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\checkval.csv', index = None, header=True)

export_csv = MissedNames.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\MissedNames.csv', index = None, header=True)

q8 = """SELECT a.MarketerContactIDText, a.MarketerName, a.AccountName FROM SubGroupByJanMay2020 a INNER JOIN Submit2019 b on (a.MarketerContactIDText =b.MarketerContactIDText);"""      

Common20192020 =  pysqldf(q8)

Common20192020.info()

con = sqlite3.connect("Common20192020.db")

Common20192020.to_sql("Common20192020", con, if_exists='replace')

q9= """SELECT * FROM Submit2019 WHERE MarketerContactIDText NOT IN (SELECT MarketerContactIDText FROM Common20192020);"""

MissedNames2020 =  pysqldf(q9)

MissedNames2020.info()

### Let's bring the AccountName in 

con = sqlite3.connect("MissedNames2020.db")

MissedNames2020.to_sql("MissedNames2020", con, if_exists='replace')

q68 = """SELECT a.AccountName, b.* FROM Common20192020 a INNER JOIN MissedNames2020 b on (a.MarketerContactIDText =b.MarketerContactIDText);"""      

MissedNames2020_add =  pysqldf(q68)




q880 = """SELECT a.*, b.Status, b.Title, b.DeletedFlag FROM MissedNames2020 a INNER JOIN MarketerlistDW b on (a.MarketerContactIDText =b.SFContactId);"""      

checkval1 =  pysqldf(q880)

export_csv = checkval1.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\checkval1.csv', index = None, header=True)


export_csv = MissedNames2020.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\MissedNames2020.csv', index = None, header=True)



