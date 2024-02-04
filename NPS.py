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

### NPS_2019 Raw Data
NPS_2019_Raw = pd.read_csv('C:/Users/test/Documents/NPS_Score/NPS_Refresh01032020/NPS_RawProcessed2019.csv',encoding= 'iso-8859-1')

NPS_2019_SuAmt = pd.read_csv('C:/Users/test/Documents/NPS_Score/NPS_Refresh01032020/NPS_SubmitAmt2019.csv',encoding= 'iso-8859-1')

NPS_2019_SuCnt = pd.read_csv('C:/Users/test/Documents/NPS_Score/NPS_Refresh01032020/NPS_SubmitCount2019.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("NPS_2019_SuAmt.db")

NPS_2019_SuAmt.to_sql("NPS_2019_SuAmt", con, if_exists='replace')

con = sqlite3.connect("NPS_2019_SuCnt.db")

NPS_2019_SuCnt.to_sql("NPS_2019_SuCnt", con, if_exists='replace')

q  = """SELECT * FROM NPS_2019_SuAmt a
        JOIN NPS_2019_SuCnt b on a.AdvisorName = b.AdvisorName1;"""
        
NPS_2019_Amt_Cont =  pysqldf(q)  

NPS_2019_Amt_Cont.info()


NPS_2019_Amt_Cont_Sel= NPS_2019_Amt_Cont[[ 'AdvisorName1','SubmitAmount','PercentSubmitAmount','CumPercentSubmitAmount','SubmitCount','PercentSubmitCount','CumPercentSubmitCount']]

con = sqlite3.connect("NPS_2019_Amt_Cont_Sel.db")

NPS_2019_Raw.info()

NPS_2019_Amt_Cont_Sel.to_sql("NPS_2019_Amt_Cont_Sel", con, if_exists='replace')

con = sqlite3.connect("NPS_2019_Raw.db")

NPS_2019_Raw.to_sql("NPS_2019_Raw", con, if_exists='replace')

NPS_2019_Raw_Sum = pd.read_sql("SELECT SC_ContactID, AdvisorName, AdvisorEmail, sum(SubmitAmount) as TotSubmitCount, sum(SubmitCount) as TotSubmitCnt FROM NPS_2019_Raw group by AdvisorName",con)

NPS_2019_Raw_Sum.info()

con = sqlite3.connect("NPS_2019_Raw_Sum.db")

NPS_2019_Raw_Sum.to_sql("NPS_2019_Raw_Sum", con, if_exists='replace')

k  = """SELECT * FROM NPS_2019_Raw_Sum a
        JOIN NPS_2019_Amt_Cont_Sel b on a.AdvisorName = b.AdvisorName1;"""
        
NPS_2019_Final1 =  pysqldf(k)  

NPS_2019_Final1.info()

NPS_2019_Final= NPS_2019_Final1[[ 'SC_ContactID','AdvisorName','AdvisorEmail','SubmitAmount','PercentSubmitAmount','CumPercentSubmitAmount','SubmitCount','PercentSubmitCount','CumPercentSubmitCount']]

NPS_2019_Final.info()
con = sqlite3.connect("NPS_2019_Final.db")

NPS_2019_Final.to_sql("NPS_2019_Final", con, if_exists='replace')

NPS_2019_Raw_BU = pd.read_sql("SELECT * FROM NPS_2019_Final where (SubmitAmount >= 1000000 and SubmitCount >=5)",con)

export_csv = NPS_2019_Final.to_csv (r'C:\Users\test\Documents\NPS_Score\NPS_Refresh01032020\Share\NPS_2019_Final.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

export_csv = NPS_2019_Raw_BU.to_csv (r'C:\Users\test\Documents\NPS_Score\NPS_Refresh01032020\Share\NPS_2019_Raw_BU.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

NPS_2019_FinalCopy= NPS_2019_Final

NPS_2019_FinalCopy.info()



export_csv = NPS_2019_Final.to_csv (r'C:\Users\test\Documents\NPS_Score\NPS_Refresh01032020\Share\NPS_2019_Final.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


### End of 2019 Analysis


### Start of 2018

### Replicate the full process for 2018

NPS_2018_Raw = pd.read_csv('C:/Users/test/Documents/NPS_Score/NPS_Refresh01032020/NPS_RawProcessed2018.csv',encoding= 'iso-8859-1')

NPS_2018_SuAmt = pd.read_csv('C:/Users/test/Documents/NPS_Score/NPS_Refresh01032020/NPS_SubmitAmt2018.csv',encoding= 'iso-8859-1')

NPS_2018_SuCnt = pd.read_csv('C:/Users/test/Documents/NPS_Score/NPS_Refresh01032020/NPS_SubmitCount2018.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("NPS_2018_SuAmt.db")

NPS_2018_SuAmt.to_sql("NPS_2018_SuAmt", con, if_exists='replace')

con = sqlite3.connect("NPS_2018_SuCnt.db")

NPS_2018_SuCnt.to_sql("NPS_2018_SuCnt", con, if_exists='replace')

p  = """SELECT * FROM NPS_2018_SuAmt a
        JOIN NPS_2018_SuCnt b on a.AdvisorName = b.AdvisorName1;"""
        
NPS_2018_Amt_Cont =  pysqldf(p)  

NPS_2018_Amt_Cont.info()


NPS_2018_Amt_Cont_Sel= NPS_2018_Amt_Cont[[ 'AdvisorName1','SubmitAmount','PercentSubmitAmount','CumPercentSubmitAmount','SubmitCount','PercentSubmitCount','CumPercentSubmitCount']]

con = sqlite3.connect("NPS_2018_Amt_Cont_Sel.db")

NPS_2018_Raw.info()

NPS_2018_Amt_Cont_Sel.to_sql("NPS_2018_Amt_Cont_Sel", con, if_exists='replace')

con = sqlite3.connect("NPS_2018_Raw.db")

NPS_2018_Raw.to_sql("NPS_2018_Raw", con, if_exists='replace')

NPS_2018_Raw_Sum = pd.read_sql("SELECT SC_ContactID, AdvisorName, AdvisorEmail, sum(SubmitAmount) as TotSubmitCount, sum(SubmitCount) as TotSubmitCnt FROM NPS_2018_Raw group by AdvisorName",con)

NPS_2018_Raw_Sum.info()

con = sqlite3.connect("NPS_2018_Raw_Sum.db")

NPS_2018_Raw_Sum.to_sql("NPS_2018_Raw_Sum", con, if_exists='replace')

kk1  = """SELECT * FROM NPS_2018_Raw_Sum a
        JOIN NPS_2018_Amt_Cont_Sel b on a.AdvisorName = b.AdvisorName1;"""
        
NPS_2018_Final1 =  pysqldf(kk1)

NPS_2018_Final1.info()

NPS_2018_Final= NPS_2018_Final1[[ 'SC_ContactID','AdvisorName','AdvisorEmail','SubmitAmount','PercentSubmitAmount','CumPercentSubmitAmount','SubmitCount','PercentSubmitCount','CumPercentSubmitCount']]

NPS_2018_FinalCopy= NPS_2018_Final
NPS_2018_FinalCopy.info()


export_csv = NPS_2018_Final.to_csv (r'C:\Users\test\Documents\NPS_Score\NPS_Refresh01032020\Share\NPS_2018_Final.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path



###******************************************************

#### Fallen Angels of #2018


## Lets find out the match between NPS_2019_Raw and NPS_2018_Raw

con = sqlite3.connect("NPS_2018_Final.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
NPS_2018_Final.to_sql("NPS_2018_Final", con, if_exists='replace')

con = sqlite3.connect("NPS_2019_Final.db")

NPS_2019_Final.to_sql("NPS_2019_Final", con, if_exists='replace')

c  = """SELECT c.SC_ContactID, c.AdvisorName, c.AdvisorEmail,c.SubmitAmount,c.PercentSubmitAmount,c.CumPercentSubmitAmount,c.SubmitCount,c.PercentSubmitCount,c.CumPercentSubmitCount  FROM NPS_2018_Final c
        INNER JOIN NPS_2019_Final d on (c.SC_ContactID = d.SC_ContactID);"""
        
Match =  pysqldf(c) 

con = sqlite3.connect("Match.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Match.to_sql("Match", con, if_exists='replace')

w2  = """SELECT * FROM NPS_2018_Final WHERE AdvisorName NOT IN (SELECT AdvisorName FROM Match);"""
        
FallenAngels =  pysqldf(w2)  

con = sqlite3.connect("FallenAngels.db")

FallenAngels.to_sql("FallenAngels", con, if_exists='replace')

FallenAngels_BU = pd.read_sql("SELECT * FROM FallenAngels where (SubmitAmount >= 1000000 and SubmitCount >=5)",con)


### Let's extract the first name and last name for Jason

NPS_2019_FinalCopy['First_Name'] = NPS_2019_FinalCopy.AdvisorName.str.split(' ', expand = True)[0]

NPS_2019_FinalCopy['Last_Name'] = NPS_2019_FinalCopy.AdvisorName.str.split(' ', expand = True)[1]

FallenAngels['First_Name'] = NPS_2018_FinalCopy.AdvisorName.str.split(' ', expand = True)[0]

FallenAngels['Last_Name'] = NPS_2018_FinalCopy.AdvisorName.str.split(' ', expand = True)[1]

export_csv = NPS_2019_Final.to_csv (r'C:\Users\test\Documents\NPS_Score\NPS_Refresh01032020\Share\NPS_2019_Final.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

export_csv = FallenAngels.to_csv (r'C:\Users\test\Documents\NPS_Score\NPS_Refresh01032020\Share\FallenAngels.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

export_csv = FallenAngels_BU.to_csv (r'C:\Users\test\Documents\NPS_Score\NPS_Refresh01032020\Share\FallenAngels_BU.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
