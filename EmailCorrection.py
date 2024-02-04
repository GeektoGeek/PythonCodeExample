import tensorflow as ts
import pandas as pd
import os
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


### NPS_2019 Raw Data
Email_2019 = pd.read_csv('C:/Users/test/Documents/EmailEngagement/EmailAnalysis-Round2/2019MarketerAnalysis/2019EmailData.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Email_2019.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Email_2019.to_sql("Email_2019", con, if_exists='replace')

Email_2019.info()

q2  = """SELECT SubscriberKey, Name, count(SendDate), sum(OpenExist), sum(ClickExist),
      sum(EngagedFinal) FROM Email_2019 group by SubscriberKey;"""
      
EmailGroupBy =  pysqldf(q2)  


Annuity_2019 = pd.read_csv('C:/Users/test/Documents/EmailEngagement/EmailAnalysis-Round2/2019MarketerAnalysis/AnnuitySubmit2019JasonShared.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Annuity_2019.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Annuity_2019.to_sql("Annuity_2019", con, if_exists='replace')

Annuity_2019.info()

q3  = """SELECT MarketerContactIDText, MarketerMarketerName, count(SubmitDate), sum(SubmitAmount)
      FROM Annuity_2019 group by MarketerContactIDText;"""
      
SubmitGroupBy =  pysqldf(q3)  

con = sqlite3.connect("EmailGroupBy.db")

EmailGroupBy.to_sql("EmailGroupBy", con, if_exists='replace')

EmailGroupBy.info()

con = sqlite3.connect("SubmitGroupBy.db")

SubmitGroupBy.to_sql("SubmitGroupBy", con, if_exists='replace')

SubmitGroupBy.info()

q4 = """SELECT * 
      FROM SubmitGroupBy LEFT JOIN EmailGroupBy on MarketerContactIDText=SubscriberKey;"""
      
SubEmail =  pysqldf(q4)  

q5 = """SELECT * 
      FROM SubmitGroupBy LEFT JOIN EmailGroupBy on ((MarketerContactIDText=SubscriberKey) or (Name=MarketerMarketerName)) ;"""
      
SubEmail_both1 =  pysqldf(q5) 

SubEmail = SubEmail.to_csv (r'C:\Users\test\Documents\NPS_Score\NPS_BoardMeeting\SubEmail.csv', index = None, header=True)

#SubEmail  = SubEmail.to_csv (r'Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\2019MarketerAnalysis\SubEmail.csv', index = None, header=True) 

SubEmail_both1 = SubEmail_both1.to_csv (r'C:\Users\test\Documents\NPS_Score\NPS_BoardMeeting\SubEmail_both1.csv', index = None, header=True)


### Let's test the correlation between Submits and Emails

Sub_Email_Marketer = pd.read_csv('C:/Users/test/Documents/EmailEngagement/EmailAnalysis-Round2/2019MarketerAnalysis/SubEmail_both2.csv',encoding= 'iso-8859-1')

Sub_Email_Marketer.info()

DF_Sub_Email_Marketer = Sub_Email_Marketer[['TotalSend','TotalOpen','TotalClick','TotalEngaged', 'SubmitAmount','SubmitCount']]

myBasicCorr = DF_Sub_Email_Marketer.corr()

sns.heatmap(myBasicCorr, annot = True, vmin = -1, vmax = 1, center = 0, cmap = 'coolwarm')

myBasicCorr1 =  DF_Sub_Email_Marketer.corr()

sns.heatmap(myBasicCorr1, annot = True)
