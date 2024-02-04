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

### Shared with Alexis

FallenAngelsEmailAppend_Share = pd.read_csv('C:/Users/test/Documents/NorthAmerica/FallenAngelsEmailAppend_Share.csv',encoding= 'iso-8859-1')

FallenAngelsEmailAppend_Share.columns = FallenAngelsEmailAppend_Share.columns.str.replace(' ', '')

con = sqlite3.connect("FallenAngelsEmailAppend_Share.db")

FallenAngelsEmailAppend_Share.to_sql("FallenAngelsEmailAppend_Share", con, if_exists='replace')

FallenAngelsEmailAppend_Share.info()

OriginalReportfromSalesForce_NPNCRD = pd.read_csv('C:/Users/test/Documents/NorthAmerica/OriginalReportfromSalesForce_NPNCRD.csv',encoding= 'iso-8859-1')

OriginalReportfromSalesForce_NPNCRD.columns = OriginalReportfromSalesForce_NPNCRD.columns.str.replace(' ', '')

OriginalReportfromSalesForce_NPNCRD.columns = OriginalReportfromSalesForce_NPNCRD.columns.str.lstrip()

OriginalReportfromSalesForce_NPNCRD.columns = OriginalReportfromSalesForce_NPNCRD.columns.str.rstrip()

OriginalReportfromSalesForce_NPNCRD.columns = OriginalReportfromSalesForce_NPNCRD.columns.str.strip()

OriginalReportfromSalesForce_NPNCRD.info()

con = sqlite3.connect("OriginalReportfromSalesForce_NPNCRD.db")

OriginalReportfromSalesForce_NPNCRD.to_sql("OriginalReportfromSalesForce_NPNCRD", con, if_exists='replace')

q3  = """SELECT a.*, b.NPN, b.AgentCRD, b.CurrentRIAAccountNewCRD, b.CurrentBDAccountCRD FROM FallenAngelsEmailAppend_Share a INNER JOIN OriginalReportfromSalesForce_NPNCRD b on (a.ID=b.ContactID18);"""

FallenAngelsEmailAppend_ShareNPN =  pysqldf(q3) 



Out4 = FallenAngelsEmailAppend_ShareNPN.to_csv(r'C:/Users/test/Documents/NorthAmerica/FallenAngelsEmailAppend_ShareNPN.csv', index = None, header=True)

FallenAngelsEmailAppend_ShareNPN1 = pd.read_csv('C:/Users/test/Documents/NorthAmerica/FallenAngelsEmailAppend_ShareNPN1.csv',encoding= 'iso-8859-1')

FallenAngelsEmailAppend_ShareNPN1.columns = FallenAngelsEmailAppend_ShareNPN1.columns.str.replace(' ', '')

con = sqlite3.connect("FallenAngelsEmailAppend_ShareNPN1.db")

FallenAngelsEmailAppend_ShareNPN1.to_sql("FallenAngelsEmailAppend_ShareNPN1", con, if_exists='replace')

SampleSize592456 = pd.read_csv('C:/Users/test/Documents/DiscoverySamplesForBrian02102021/SampleSize_592456_A.csv',encoding= 'iso-8859-1')

SampleSize592456.columns = SampleSize592456.columns.str.replace(' ', '')

SampleSize592456.columns = SampleSize592456.columns.str.lstrip()

SampleSize592456.columns = SampleSize592456.columns.str.rstrip()

SampleSize592456.columns = SampleSize592456.columns.str.strip()

SampleSize592456.info()

con = sqlite3.connect("SampleSize592456.db")

SampleSize592456.to_sql("SampleSize592456", con, if_exists='replace')

q3  = """SELECT a.*, b.EmailPersonalType, b.BranchContactEmail, b.EmailBusinessType, b.EmailBusiness2Type, b.BranchPhone, b.HomePhone, b.NPN, a.AgentCRD, a.CurrentRIAAccountNewCRD, a.CurrentBDAccountCRD FROM FallenAngelsEmailAppend_ShareNPN1 a LEFT JOIN SampleSize592456 b on ((a.NPN=b.NPN));"""

FallenAngelsEmailAppend_ShareNPN =  pysqldf(q3) 

Out4 = FallenAngelsEmailAppend_ShareNPN.to_csv(r'C:/Users/test/Documents/NorthAmerica/FallenAngelsEmailAppend_ShareNPN.csv', index = None, header=True)


##q3  = """SELECT a.*, b.EmailPersonalType, b.NPN, a.AgentCRD, a.CurrentRIAAccountNewCRD, a.CurrentBDAccountCRD FROM FallenAngelsEmailAppend_ShareNPN a INNER JOIN SampleSize592456 b on ((a.NPN=b.NPN) or (a.AgentCRD=b.RepCRD));"""

##FallenAngelsEmailAppend_ShareNPN =  pysqldf(q3) 


List_ShareAlexis = pd.read_csv('C:/Users/test/Documents/Alexis_LinkedIn/List_ShareAlexis.csv',encoding= 'iso-8859-1')

List_ShareAlexis.columns = List_ShareAlexis.columns.str.replace(' ', '')

con = sqlite3.connect("List_ShareAlexis.db")

List_ShareAlexis.to_sql("List_ShareAlexis", con, if_exists='replace')


q3  = """SELECT a.*, b.EmailPersonalType, b.BranchContactEmail, b.EmailBusinessType, b.EmailBusiness2Type, b.BranchPhone, b.HomePhone, b.NPN, a.AgentCRD, a.CurrentRIAAccountNewCRD, a.CurrentBDAccountCRD FROM List_ShareAlexis a LEFT JOIN SampleSize592456 b on ((a.NPN=b.NPN));"""

List_ShareAlexis_ShareNPN =  pysqldf(q3) 









