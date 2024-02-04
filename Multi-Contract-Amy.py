
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

### Email Data

Mul_DF = pd.read_csv(r'C:\Users\test\Documents\Amy\IllusstationDataNov2018Nov2019FullData.csv',encoding= 'iso-8859-1')


con = sqlite3.connect("Mul_DF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Mul_DF.to_sql("Mul_DF", con, if_exists='replace')

Mul_DF.info()



#### Last 60 days Advisors and Income Rider

Mul_DF_60days = pd.read_sql("SELECT * FROM Mul_DF where PreparationDate ",con)






MulContract_DF = pd.read_sql("SELECT IllustrationID, count(IllusContractID) as TotalContract FROM Mul_DF group by IllustrationID order by count(IllusContractID) desc",con)

export_csv = MulContract_DF.to_csv (r'C:\Users\test\Documents\Amy\MulContract_DF.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

MulContract_DF.info()

con = sqlite3.connect("MulContract_DF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
MulContract_DF.to_sql("MulContract_DF", con, if_exists='replace')

MulContract_DFCount = pd.read_sql("SELECT distinct(IllustrationID), TotalContract FROM MulContract_DF where TotalContract > 1 ",con)

con = sqlite3.connect("MulContract_DFCount.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
MulContract_DFCount.to_sql("MulContract_DFCount", con, if_exists='replace')

q2  = """SELECT c.IllustrationID, c.TotalContract, d.Index1, d.Rider FROM MulContract_DFCount c
        INNER JOIN Mul_DF d on (c.IllustrationID = d.IllustrationID);"""
        
FinalDF =  pysqldf(q2)  

con = sqlite3.connect("FinalDF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
FinalDF.to_sql("FinalDF", con, if_exists='replace')


Index_Count  = pd.read_sql("SELECT count((Index1)), Index1, IllustrationID from FinalDF group by Index1,IllustrationID", con)

Index_Count1  = pd.read_sql("SELECT count(distinct(Index1)), Index1, IllustrationID from FinalDF group by Index1,IllustrationID", con)

export_csv = FinalDF.to_csv (r'C:\Users\test\Documents\Amy\FinalDF.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

export_csv = Index_Count.to_csv (r'C:\Users\test\Documents\Amy\Index_Count.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

#### Take the dataset of FinalDF create a pivot between IllustrationID (which are only multicontract perceived from the FinalDF and a pivot of Index1)

IndexFiguredout_DF = pd.read_csv(r'C:\Users\test\Documents\Amy\IndexFiguredOut.csv',encoding= 'iso-8859-1')

IndexFiguredout_DF.info()

### Keep a copy of the original dataset

IndexFiguredout_DFCopy= IndexFiguredout_DF

### Replace all nan to 0
IndexFiguredout_DF = IndexFiguredout_DF.replace(np.nan, 0)

con = sqlite3.connect("IndexFiguredout_DF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
IndexFiguredout_DF.to_sql("IndexFiguredout_DF", con, if_exists='replace')

IndexFiguredout_DF.info()

### How many of those are selecting one index

OneIndex_Count  = pd.read_sql("SELECT * from IndexFiguredout_DF where ((JPM_Mozaic_II = GrandTotal) or (UBS_Zebra = GrandTotal) or (SAndP_500=GrandTotal) or (EAFE=GrandTotal)) group by Illustration_ID ", con)

export_csv = OneIndex_Count.to_csv (r'C:\Users\test\Documents\Amy\OneIndex_Count.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


### IncomeRider Check

MulContract_DFCount.to_sql("MulContract_DFCount", con, if_exists='replace')

q5  = """SELECT count(d.Rider), c.illustrationID FROM MulContract_DFCount c
        Inner JOIN Mul_DF d on (c.IllustrationID = d.IllustrationID) where d.Rider!= 'No Rider' group by c.IllustrationID;"""
        
FinalIncomeRider =  pysqldf(q5)  

q6  = """SELECT count(d.Rider), c.illustrationID FROM MulContract_DFCount c
        Inner JOIN Mul_DF d on (c.IllustrationID = d.IllustrationID) where ((d.Rider= 'GLWB') or (d.Rider= 'GLWB with Bonus'))
 group by c.IllustrationID;"""
        
FinalIncomeRider1 =  pysqldf(q6)  

con = sqlite3.connect("FinalIncomeRider1.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Mul_DF.to_sql("FinalIncomeRider1", con, if_exists='replace')

### Staring income in different years 

### This has to do with the start age and end age

Sur_DF =  pd.read_csv(r'C:\Users\test\Documents\Amy\IllusstationDataNov2018Nov2019FullDataFreeSurrendarPeriodmodi.csv',encoding= 'iso-8859-1')
      
Sur_DF.info()
con = sqlite3.connect("Sur_DF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Sur_DF.to_sql("Sur_DF", con, if_exists='replace')

q7  = """SELECT c.illustrationID, d.StartAge, d.EndAge FROM FinalIncomeRider1 c
        Left JOIN Sur_DF d on (c.IllustrationID = d.IllustrationID) where ((d.StartAge !='Null') or (d.EndAge!= 'Null'))
 group by c.IllustrationID;"""

FinalIR_DiffYear =  pysqldf(q7)  

Sur1_DF =  pd.read_csv(r'C:\Users\test\Documents\Amy\Joinofthree_multicontract1.csv',encoding= 'iso-8859-1')
      
Sur1_DF.info()
con = sqlite3.connect("Sur1_DF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Sur1_DF.to_sql("Sur_DF", con, if_exists='replace')

q8  = """SELECT c.illustrationID, d.StartAge, d.EndAge FROM FinalIncomeRider1 c
        Left JOIN Sur1_DF d on (c.IllustrationID = d.IllustrationID) where ((d.StartAge !='Null') or (d.EndAge!= 'Null'))
 group by c.IllustrationID;"""

FinalIR_DiffYear2 =  pysqldf(q8)  


#### Last 60 days Advisors and Income Rider

Mul_DF = pd.read_csv(r'C:\Users\test\Documents\Amy\IllusstationDataNov2018Nov2019FullData.csv',encoding= 'iso-8859-1')


