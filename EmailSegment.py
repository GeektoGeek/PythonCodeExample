
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

### NPS_2019 Raw Data

SubmitDataRaw= pd.read_csv('C:/Users/test/Documents/EmailSegmentsForJason/SubmitDataRaw.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("SubmitDataRaw.db")

SubmitDataRaw.to_sql("SubmitDataRaw", con, if_exists='replace')
NWApp = pd.read_csv('C:/Users/test/Documents/EmailSegmentsForJason/NWAppointments.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("NWApp.db")

NWApp.to_sql("NWApp", con, if_exists='replace')

SubmitDataRaw.info()

NWApp.info()


qqq1  = """SELECT a.* FROM SubmitDataRaw a INNER JOIN NWApp b on a.AdvisorContactIDText = b.ContactID18 ;"""
        
ValidationApp =  pysqldf(qqq1) 

export_csv1 = ValidationApp.to_csv(r'C:\Users\test\Documents\EmailSegmentsForJason\ValidationApp.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path 

SuAmt2 = pd.read_csv('C:/Users/test/Documents/EmailSegmentsForJason/SubmitAmtCalc2.csv',encoding= 'iso-8859-1')

SuCnt2 = pd.read_csv('C:/Users/test/Documents/EmailSegmentsForJason/SubmitCountCalc2.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("SuAmt2.db")

SuAmt2.to_sql("SuAmt2", con, if_exists='replace')

con = sqlite3.connect("SuCnt2.db")

SuCnt2.to_sql("SuCnt2", con, if_exists='replace')

SuAmt2.info()

SuCnt2.info()

qqq  = """SELECT a.AdvisorContactIDText, a.SubmitAmt, a.AmtPercent, a.AmtCumPercent, b.SubmitCount, b.CountPercent, b.CountCumPercent FROM SuAmt2 a
        LEFT JOIN SuCnt2 b on a.AdvisorContactIDText = b.AdvisorContactIDText;"""
        
FinalSubmitAmtCount2 =  pysqldf(qqq) 

export_csv1 = FinalSubmitAmtCount2.to_csv(r'C:\Users\test\Documents\EmailSegmentsForJason\FinalSubmitAmtCount2.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path 


con = sqlite3.connect("FinalSubmitAmtCount2.db")

FinalSubmitAmtCount2.to_sql("FinalSubmitAmtCount2", con, if_exists='replace')

NWApp.info()

qqq12  = """SELECT a.* FROM FinalSubmitAmtCount2 a INNER JOIN NWApp b on a.AdvisorContactIDText = b.ContactID18 ;"""
        
ValidationApp1 =  pysqldf(qqq12) 


export_csv1 = FinalSubmitAmtCount.to_csv(r'C:\Users\test\Documents\EmailSegmentsForJason\FinalSubmitAmtCount.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path 

NPS_2019_Amt_Cont.info()

