
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

## Dataset 1

IllusDF = pd.read_csv('C:/Users/test/Documents/IllustrationAnalysis/Data_Illustration_IllustrationContracts09042019.csv',encoding= 'iso-8859-1')

##IllusDF.columns= AnnuitySub3YrsMar.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

IllusDF.info()

IllusDF['IncomePercentage.1'].describe()

IllusDF['PreparationDate'] = pd.to_datetime(IllusDF['PreparationDate'])

### Create the database
con = sqlite3.connect("IllusDF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
IllusDF.to_sql("IllusDF", con, if_exists='replace')

### Producers(Advissors) information: let's cut the data based on 3 separate years first

PrepCount = pd.read_sql("SELECT count(PreparationDate) as PrepCount, PreparationDate FROM IllusDF group by PreparationDate",con)

### Let's take all the names I received from Megan

AnnEmp_Illus = pd.read_csv('C:/Users/test/Documents/IllustrationAnalysis/IllusByAnnexus.csv',encoding= 'iso-8859-1')

### Create the database
con = sqlite3.connect("AnnEmp_Illus.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
AnnEmp_Illus.to_sql("AnnEmp_Illus", con, if_exists='replace')

q  = """SELECT * FROM IllusDF 
        EXCEPT 
        SELECT a.* FROM IllusDF a 
        JOIN AnnEmp_Illus b where ((a.ClientFirstName = b.ClientFirstName) and (a.ClientLastName = b.ClientLastName));"""
                
Illus_Clean =  pysqldf(q)        

### This seems worked fine as the # of rows reduced from 516,866 to 516,439

### Now export this to a csv and work on this...

exp = Illus_Clean.to_csv (r'C:\Users\test\Documents\IllustrationAnalysis\Illus_Clean.csv', index = None, header=True) 

                                                           






