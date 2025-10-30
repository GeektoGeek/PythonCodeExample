# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:33:25 2020

@author: dsarkar
"""


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

# This line tells the notebook to show plots inside of the notebook

sess = ts.Session()
a = ts.constant(10)
b = ts.constant(32)
print(sess.run(a+b))


cloverU = pd.read_csv('c.csv')

sliceU = pd.read_csv('C:/BusinessDevelopment/LeftysPizzaBreakfast/contactssliceU.csv')

sliceU['Phone1'] = sliceU['Phone1'].astype(str)

cloverU['Phone1'] = cloverU['Phone1'].astype(str)

con = sqlite3.connect("cloverU.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
cloverU.to_sql("cloverU", con, if_exists='replace')

con = sqlite3.connect("sliceU.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
sliceU.to_sql("sliceU", con, if_exists='replace')

qqq122  = """SELECT a.* FROM cloverU a INNER JOIN sliceU b on a.Phone1 = b.Phone1;"""

Common=  pysqldf(qqq122) 

cloverU.info()

sliceU.info()

Out5 = cloverU.to_csv(r'C:\BusinessDevelopment\LeftysPizzaBreakfast\cloverU.csv', index = None, header=True)

Out5 = sliceU.to_csv(r'C:\BusinessDevelopment\LeftysPizzaBreakfast\sliceU.csv', index = None, header=True)



# Perform an inner join on 'key_column'
inner_join = pd.merge(cloverU, sliceU, on='Phone1', how='inner')

overlap = pd.merge(cloverU, sliceU, on='Phone1')

# Perform an outer merge to get all rows from both dataframes

merged = pd.merge(sliceU, cloverU, on='Phone1', how='outer', indicator=True)

# Filter out the overlapping rows
unique = merged[merged['_merge'] != 'both']

# Drop the '_merge' column as it's no longer needed
unique = unique.drop(columns=['_merge'])

