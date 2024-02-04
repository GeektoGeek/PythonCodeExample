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

### Athene Coolers

### Load the Cooler list

### AugustNYC2019 Data

## First process the in QuestionText columns

FinalListcamefromTony = pd.read_csv('C:/Users/test/Documents/NorthAmerica/FinalListcamefromTony.csv',encoding= 'iso-8859-1')

FinalListcamefromTony.columns = FinalListcamefromTony.columns.str.replace(' ', '')

FinalListcamefromTony.info()

FinalList_SFEmail = FinalListcamefromTony[["ID", "FirstName","LastName","SFPhone","Phone1","Phone2_OtherPhone","Phone3_HomePhone","Zip1_HomeZip","Zip2_MailingZip","Country","SFEmail"]]

FinalList_SFEmail1= FinalList_SFEmail.loc[pd.notnull(FinalList_SFEmail.SFEmail)]

FinalListcamefromTony.info()

FinalList_SFAlternateEmail = FinalListcamefromTony[["ID", "FirstName","LastName","SFPhone","Phone1","Phone2_OtherPhone","Phone3_HomePhone","Zip1_HomeZip","Zip2_MailingZip","Country","SFAlternateEmail"]]

FinalList_SFAlternateEmail1= FinalList_SFAlternateEmail.loc[pd.notnull(FinalList_SFAlternateEmail.SFAlternateEmail)]

FinalListcamefromTony.info()

FinalList_Email1_NationWide_Email = FinalListcamefromTony[["ID", "FirstName","LastName","SFPhone","Phone1","Phone2_OtherPhone","Phone3_HomePhone","Zip1_HomeZip","Zip2_MailingZip","Country","Email1_NationWide_Email"]]

FinalList_Email1_NationWide_Email1= FinalList_Email1_NationWide_Email.loc[pd.notnull(FinalList_Email1_NationWide_Email.Email1_NationWide_Email)]

FinalListcamefromTony.info()

FinalList_Email2_AlternateEmail = FinalListcamefromTony[["ID", "FirstName","LastName","SFPhone","Phone1","Phone2_OtherPhone","Phone3_HomePhone","Zip1_HomeZip","Zip2_MailingZip","Country","Email2_AlternateEmail"]]

FinalList_Email2_AlternateEmail1= FinalList_Email2_AlternateEmail.loc[pd.notnull(FinalList_Email2_AlternateEmail.Email2_AlternateEmail)]

FinalListcamefromTony.info()

FinalList_Email3_FirstAtheneAdvisor = FinalListcamefromTony[["ID", "FirstName","LastName","SFPhone","Phone1","Phone2_OtherPhone","Phone3_HomePhone","Zip1_HomeZip","Zip2_MailingZip","Country","Email3_FirstAtheneAdvisor"]]

FinalList_Email3_FirstAtheneAdvisor1= FinalList_Email3_FirstAtheneAdvisor.loc[pd.notnull(FinalList_Email3_FirstAtheneAdvisor.Email3_FirstAtheneAdvisor)]

FinalListcamefromTony.info()

FinalList_Email4_AtheneEmail = FinalListcamefromTony[["ID", "FirstName","LastName","SFPhone","Phone1","Phone2_OtherPhone","Phone3_HomePhone","Zip1_HomeZip","Zip2_MailingZip","Country","Email4_AtheneEmail"]]

FinalList_Email4_AtheneEmail1= FinalList_Email4_AtheneEmail.loc[pd.notnull(FinalList_Email4_AtheneEmail.Email4_AtheneEmail)]

FinalListcamefromTony.info()

FinalList_Email5_MinnLifeEmail = FinalListcamefromTony[["ID", "FirstName","LastName","SFPhone","Phone1","Phone2_OtherPhone","Phone3_HomePhone","Zip1_HomeZip","Zip2_MailingZip","Country","Email5_MinnLifeEmail"]]

FinalList_Email5_MinnLifeEmail1= FinalList_Email5_MinnLifeEmail.loc[pd.notnull(FinalList_Email5_MinnLifeEmail.Email5_MinnLifeEmail)]

### Lets take SFEmail and SFAlternateEmail

## Approach one


FinalList_SFEmail1['Email'] = FinalList_SFEmail1['SFEmail']

FinalList_SFAlternateEmail1['Email'] = FinalList_SFAlternateEmail1['SFAlternateEmail']


con = sqlite3.connect("FinalList_SFEmail1.db")

FinalList_SFEmail1.to_sql("FinalList_SFEmail1", con, if_exists='replace')

FinalList_SFEmail1.info()

con = sqlite3.connect("FinalList_SFAlternateEmail1.db")

FinalList_SFAlternateEmail1.to_sql("FinalList_SFAlternateEmail1", con, if_exists='replace')

FinalList_SFAlternateEmail1.info()

q3  = """SELECT a.* FROM FinalList_SFEmail1 a INNER JOIN FinalList_SFAlternateEmail1 b on (a.Email =b.Email);"""

CommonEmail_AltEmail =  pysqldf(q3) 

p23  = """SELECT * FROM FinalList_SFAlternateEmail1 WHERE Email NOT IN (SELECT Email FROM CommonEmail_AltEmail);"""
        
CommonAltEmailExclusive=  pysqldf(p23)

Check333= pd.merge(FinalList_SFEmail1, CommonAltEmailExclusive, on="Email", how='outer')

Out4 = Check333.to_csv(r'C:/Users/test/Documents/NorthAmerica/Check333.csv', index = None, header=True)

### After out you have to do manual processing of check333 to create one email column

SFEmail_AltEmail = pd.read_csv('C:/Users/test/Documents/NorthAmerica/Check_SFEmail_SFAlternateEmail.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("SFEmail_AltEmail.db")

SFEmail_AltEmail.to_sql("SFEmail_AltEmail", con, if_exists='replace')

FinalList_Email1_NationWide_Email1.info()

FinalList_Email1_NationWide_Email1['Email'] = FinalList_Email1_NationWide_Email1['Email1_NationWide_Email']

FinalList_Email1_NationWide_Email1.info()

con = sqlite3.connect("FinalList_Email1_NationWide_Email1.db")

FinalList_Email1_NationWide_Email1.to_sql("FinalList_Email1_NationWide_Email1", con, if_exists='replace')

FinalList_Email1_NationWide_Email1.info()

FinalList_Email3_FirstAtheneAdvisor1.info()

FinalList_Email3_FirstAtheneAdvisor1['Email'] = FinalList_Email3_FirstAtheneAdvisor1['Email3_FirstAtheneAdvisor']

con = sqlite3.connect("FinalList_Email3_FirstAtheneAdvisor1.db")

FinalList_Email3_FirstAtheneAdvisor1.to_sql("FinalList_Email3_FirstAtheneAdvisor1", con, if_exists='replace')

q3  = """SELECT a.* FROM FinalList_Email1_NationWide_Email1 a INNER JOIN FinalList_Email3_FirstAtheneAdvisor1 b on (a.Email =b.Email);"""

CommonNW_Athene =  pysqldf(q3) 

p23  = """SELECT * FROM FinalList_Email3_FirstAtheneAdvisor1 WHERE Email NOT IN (SELECT Email FROM CommonNW_Athene);"""
        
CommonAtheneExclusive=  pysqldf(p23)

### Validation same ComonNW_Athene single step
q33  = """SELECT a.* FROM FinalList_Email3_FirstAtheneAdvisor1 a LEFT JOIN FinalList_Email1_NationWide_Email1 b on (a.Email =b.Email) where b.Email is NULL;"""

CommonAtheneExclusive1 =  pysqldf(q33) 

Check444= pd.merge(FinalList_Email1_NationWide_Email1, CommonAtheneExclusive, on="Email", how='outer')

Out4 = Check444.to_csv(r'C:/Users/test/Documents/NorthAmerica/Check444.csv', index = None, header=True)

#### 

con = sqlite3.connect("FinalList_Email4_AtheneEmail1.db")

FinalList_Email4_AtheneEmail1.to_sql("FinalList_Email4_AtheneEmail1", con, if_exists='replace')

FinalList_Email4_AtheneEmail1.info()

FinalList_Email4_AtheneEmail1['Email'] = FinalList_Email4_AtheneEmail1['Email4_AtheneEmail']

con = sqlite3.connect("FinalList_Email5_MinnLifeEmail1.db")

FinalList_Email5_MinnLifeEmail1.to_sql("FinalList_Email5_MinnLifeEmail1", con, if_exists='replace')

FinalList_Email5_MinnLifeEmail1.info()

FinalList_Email5_MinnLifeEmail1['Email'] = FinalList_Email5_MinnLifeEmail1['Email5_MinnLifeEmail']

q34  = """SELECT a.* FROM FinalList_Email4_AtheneEmail1 a LEFT JOIN FinalList_Email5_MinnLifeEmail1 b on (a.Email =b.Email) where b.Email is NULL;"""

CommonAtheneExclusive2 =  pysqldf(q34) 

Check555= pd.merge(FinalList_Email5_MinnLifeEmail1, CommonAtheneExclusive2, on="Email", how='outer')

Out4 = Check555.to_csv(r'C:/Users/test/Documents/NorthAmerica/Check555.csv', index = None, header=True)

### Bring the three tables

Check_SFEmail_SFAlternateEmail = pd.read_csv('C:/Users/test/Documents/NorthAmerica/Check_SFEmail_SFAlternateEmail.csv',encoding= 'iso-8859-1')

CheckNW_AtheneEmail = pd.read_csv('C:/Users/test/Documents/NorthAmerica/CheckNW_AtheneEmail.csv',encoding= 'iso-8859-1')

CheckAthene_MinnLife = pd.read_csv('C:/Users/test/Documents/NorthAmerica/CheckAthene_MinnLife.csv',encoding= 'iso-8859-1')

Check666= pd.merge(Check_SFEmail_SFAlternateEmail, CheckNW_AtheneEmail, on="Email", how='outer')

Out4 = Check666.to_csv(r'C:/Users/test/Documents/NorthAmerica/Check666.csv', index = None, header=True)

Final666 = pd.read_csv('C:/Users/test/Documents/NorthAmerica/Final666.csv',encoding= 'iso-8859-1')

Check777= pd.merge(Final666, CheckAthene_MinnLife, on="Email", how='outer')

Out4 = Check777.to_csv(r'C:/Users/test/Documents/NorthAmerica/Check777.csv', index = None, header=True)

### Once you export it, you can still need to bring the data into together...

## Align it in the same row etc which is Final777

Final777 = pd.read_csv('C:/Users/test/Documents/NorthAmerica/Final777.csv',encoding= 'iso-8859-1')

Final777.info()


df4 = pd.DataFrame(Final777,columns=['ID_x_x', 'FirstName_x_x','LastName_x_x', 'SFPhone_x_x','Phone1_x_x','Phone2_OtherPhone_x_x','Phone3_HomePhone_x_x','Zip1_HomeZip_x_x','Zip2_MailingZip_x_x','Country_x_x','Email'])

df5= df4.drop_duplicates()

df5 = df5.reset_index()

Out4 = df5.to_csv(r'C:/Users/test/Documents/NorthAmerica/df5.csv', index = None, header=True)


query = """
SELECT s.*, m.*
FROM   FinalList_SFEmail1 s
       LEFT JOIN FinalList_SFAlternateEmail1 m
          ON s.SFEmail = m.SFAlternateEmail
UNION All
SELECT s.*, m.*
FROM   FinalList_SFEmail1 m
       LEFT JOIN FinalList_SFAlternateEmail1 s
          ON s.SFAlternateEmail = m.SFEmail
WHERE  s.SFAlternateEmail IS NULL;
"""
Check111=  pysqldf(query)

FinalList_SFEmail1['Email'] = FinalList_SFEmail1['SFEmail']

FinalList_SFAlternateEmail1['Email'] = FinalList_SFAlternateEmail1['SFAlternateEmail']

Check222= pd.merge(FinalList_SFEmail1, FinalList_SFAlternateEmail1, on="Email", how='outer')
