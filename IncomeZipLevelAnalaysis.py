# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:42:17 2020

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

###Bring the Submit Data


SubmitData2020 = pd.read_csv('C:/Users/DSarkar/Documents/ZipLevelIncomeAnalysis/2020SubmitData.csv',encoding= 'iso-8859-1')

SubmitData2020.columns = SubmitData2020.columns.str.replace(' ', '')

SubmitData2020.columns = SubmitData2020.columns.str.lstrip()

SubmitData2020.columns = SubmitData2020.columns.str.rstrip()

SubmitData2020.columns = SubmitData2020.columns.str.strip()

SubmitData2020['SubmitDate'] = pd.to_datetime(SubmitData2020['SubmitDate'])

con = sqlite3.connect("SubmitData2020.db")

SubmitData2020.to_sql("SubmitData2020", con, if_exists='replace')

SubmitData2020.info()

q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvisorCount, count(SubmitDate) as SubmitCountnt, sum(SubmitAmount) as SubmitAmount, 
      MailingZipU FROM SubmitData2020 group by MailingZipU;"""
      
SubmitDataGrByZip1 =  pysqldf(q3) 

Out4 = SubmitDataGrByZip1.to_csv(r'C:/Users/DSarkar/Documents/ZipLevelIncomeAnalysis/SubmitDataGrByZip1.csv', index = None, header=True)

###Zip Code Income Data

HosuseholdIncomeZipData = pd.read_csv('C:/Users/DSarkar/Documents/ZipLevelIncomeAnalysis/Median_Household_Income_by_ZipcodeUMichModi.csv',encoding= 'iso-8859-1')

HosuseholdIncomeZipData.columns = HosuseholdIncomeZipData.columns.str.replace(' ', '')

HosuseholdIncomeZipData.columns = HosuseholdIncomeZipData.columns.str.lstrip()

HosuseholdIncomeZipData.columns = HosuseholdIncomeZipData.columns.str.rstrip()

HosuseholdIncomeZipData.columns = HosuseholdIncomeZipData.columns.str.strip()

con = sqlite3.connect("HosuseholdIncomeZipData.db")

HosuseholdIncomeZipData.to_sql("HosuseholdIncomeZipData", con, if_exists='replace')

HosuseholdIncomeZipData.info()

SubmitDataGrByZip1modi1 = pd.read_csv('C:/Users/DSarkar/Documents/ZipLevelIncomeAnalysis/SubmitDataGrByZip1modi1.csv',encoding= 'iso-8859-1')

SubmitDataGrByZip1modi1.columns = SubmitDataGrByZip1modi1.columns.str.replace(' ', '')

SubmitDataGrByZip1modi1.columns = SubmitDataGrByZip1modi1.columns.str.lstrip()

SubmitDataGrByZip1modi1.columns = SubmitDataGrByZip1modi1.columns.str.rstrip()

SubmitDataGrByZip1modi1.columns = SubmitDataGrByZip1modi1.columns.str.strip()

con = sqlite3.connect("SubmitDataGrByZip1modi1.db")

SubmitDataGrByZip1modi1.to_sql("SubmitDataGrByZip1modi1", con, if_exists='replace')

SubmitDataGrByZip1modi1.info()

vm11 = """SELECT a.*, b.* FROM HosuseholdIncomeZipData a INNER JOIN SubmitDataGrByZip1modi1 b on ((a.Zip=b.MailingZipU ));"""      

CommonMatchZip1modi1Submit=  pysqldf(vm11)

Out4 = CommonMatchZip1modi1Submit.to_csv(r'C:/Users/DSarkar/Documents/ZipLevelIncomeAnalysis/CommonMatchZip1modi1Submit.csv', index = None, header=True)

### How many top 50 and top 100 zip codes from Income perspective where Annexus is present?

##top 50
HosuseholdIncomeZipData.info()

df1= HosuseholdIncomeZipData.nlargest(50,'MedianIncome')

out = df1.to_csv(r'C:/Users/DSarkar/Documents/ZipLevelIncomeAnalysis/df1.csv', index = None, header=True)


con = sqlite3.connect("df1.db")

df1.to_sql("df1", con, if_exists='replace')


vm11 = """SELECT a.*, b.* FROM df1 a INNER JOIN SubmitDataGrByZip1modi1 b on ((a.Zip=b.MailingZipU ));"""      

Common50match=  pysqldf(vm11)

out = Common50match.to_csv(r'C:/Users/DSarkar/Documents/ZipLevelIncomeAnalysis/Common50match.csv', index = None, header=True)

##top 100

HosuseholdIncomeZipData.info()

df2= HosuseholdIncomeZipData.nlargest(100,'MedianIncome')

out = df2.to_csv(r'C:/Users/DSarkar/Documents/ZipLevelIncomeAnalysis/df2.csv', index = None, header=True)


con = sqlite3.connect("df2.db")

df2.to_sql("df2", con, if_exists='replace')


vm11 = """SELECT a.*, b.* FROM df2 a INNER JOIN SubmitDataGrByZip1modi1 b on ((a.Zip=b.MailingZipU ));"""      

Common100match=  pysqldf(vm11)

out = Common100match.to_csv(r'C:/Users/DSarkar/Documents/ZipLevelIncomeAnalysis/Common100match.csv', index = None, header=True)

## how many top 100 Annexus Submitted business also included in

SubmitDataGrByZip1modi1.info() 

df2Submit= SubmitDataGrByZip1modi1.nlargest(100,'SubmitAmount')


vm11 = """SELECT a.*, b.* FROM df2 a INNER JOIN df2Submit b on ((a.Zip=b.MailingZipU ));"""      

Common100_match_match=  pysqldf(vm11)


