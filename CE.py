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
##import featuretools as ft
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

#### 2022 Data based on Home State

SubmittedAdv_2022 = pd.read_csv('C:/Users/test/Documents/SuperCE/SubmittedAdv_2022.csv',encoding= 'iso-8859-1')

SubmittedAdv_2022.columns = SubmittedAdv_2022.columns.str.replace(' ', '')

SubmittedAdv_2022.info()

q3  = """SELECT count(AdvisorContactIDText), AdvisorContactHomeStateProvince FROM SubmittedAdv_2022 group by AdvisorContactHomeStateProvince;"""

State_AdvCount =  pysqldf(q3)

###FL

q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvCount, sum(SubmitAmount) as Submits, count(SubmitDate) as AppCount, AdvisorContactHomeStateProvince,AdvisorContactHomeCity FROM SubmittedAdv_2022 where AdvisorContactHomeStateProvince='FL' group by AdvisorContactHomeStateProvince,AdvisorContactHomeCity order by AdvisorContactHomeCity, sum(SubmitAmount) desc ;"""

FL_AdvCount_City =  pysqldf(q3)

FL_AdvCount_City.info()

FL_AdvCount_City['AdvisorContactHomeCity'] = FL_AdvCount_City['AdvisorContactHomeCity'].str.upper()

FL_AdvCount_City1=FL_AdvCount_City.groupby("AdvisorContactHomeCity", as_index=False).agg(AdvCount=("AdvCount", "sum"), Submits=("Submits", "sum"),AppCount=("AppCount", "sum"))

out = FL_AdvCount_City1.to_csv (r'C:\Users\test\Documents\SuperCE\FL_AdvCount_City1.csv', index = None, header=True)

## CA 
q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvCount, sum(SubmitAmount) as Submits, count(SubmitDate) as AppCount, AdvisorContactHomeStateProvince,AdvisorContactHomeCity FROM SubmittedAdv_2022 where AdvisorContactHomeStateProvince='CA' group by AdvisorContactHomeStateProvince,AdvisorContactHomeCity order by AdvisorContactHomeCity, sum(SubmitAmount) desc ;"""

CA_AdvCount_City =  pysqldf(q3)

CA_AdvCount_City.info()

CA_AdvCount_City['AdvisorContactHomeCity'] = CA_AdvCount_City['AdvisorContactHomeCity'].str.upper()

CA_AdvCount_City1=CA_AdvCount_City.groupby("AdvisorContactHomeCity", as_index=False).agg(AdvCount=("AdvCount", "sum"), Submits=("Submits", "sum"),AppCount=("AppCount", "sum"))

out = CA_AdvCount_City1.to_csv (r'C:\Users\test\Documents\SuperCE\CA_AdvCount_City1.csv', index = None, header=True)

## TX
q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvCount, sum(SubmitAmount) as Submits, count(SubmitDate) as AppCount, AdvisorContactHomeStateProvince,AdvisorContactHomeCity FROM SubmittedAdv_2022 where AdvisorContactHomeStateProvince='TX' group by AdvisorContactHomeStateProvince,AdvisorContactHomeCity order by AdvisorContactHomeCity, sum(SubmitAmount) desc ;"""

TX_AdvCount_City =  pysqldf(q3)

TX_AdvCount_City.info()

TX_AdvCount_City['AdvisorContactHomeCity'] = TX_AdvCount_City['AdvisorContactHomeCity'].str.upper()

TX_AdvCount_City1=TX_AdvCount_City.groupby("AdvisorContactHomeCity", as_index=False).agg(AdvCount=("AdvCount", "sum"), Submits=("Submits", "sum"),AppCount=("AppCount", "sum"))

out = TX_AdvCount_City1.to_csv (r'C:\Users\test\Documents\SuperCE\TX_AdvCount_City1.csv', index = None, header=True)

## AZ
q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvCount, sum(SubmitAmount) as Submits, count(SubmitDate) as AppCount, AdvisorContactHomeStateProvince,AdvisorContactHomeCity FROM SubmittedAdv_2022 where AdvisorContactHomeStateProvince='AZ' group by AdvisorContactHomeStateProvince,AdvisorContactHomeCity order by AdvisorContactHomeCity, sum(SubmitAmount) desc ;"""

AZ_AdvCount_City =  pysqldf(q3)

AZ_AdvCount_City.info()

AZ_AdvCount_City['AdvisorContactHomeCity'] = AZ_AdvCount_City['AdvisorContactHomeCity'].str.upper()

AZ_AdvCount_City1=AZ_AdvCount_City.groupby("AdvisorContactHomeCity", as_index=False).agg(AdvCount=("AdvCount", "sum"), Submits=("Submits", "sum"),AppCount=("AppCount", "sum"))

out = AZ_AdvCount_City1.to_csv (r'C:\Users\test\Documents\SuperCE\AZ_AdvCount_City1.csv', index = None, header=True)

## NC

q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvCount, sum(SubmitAmount) as Submits, count(SubmitDate) as AppCount, AdvisorContactHomeStateProvince,AdvisorContactHomeCity FROM SubmittedAdv_2022 where AdvisorContactHomeStateProvince='NC' group by AdvisorContactHomeStateProvince,AdvisorContactHomeCity order by AdvisorContactHomeCity, sum(SubmitAmount) desc ;"""

NC_AdvCount_City =  pysqldf(q3)

NC_AdvCount_City.info()

NC_AdvCount_City['AdvisorContactHomeCity'] = NC_AdvCount_City['AdvisorContactHomeCity'].str.upper()

NC_AdvCount_City1=NC_AdvCount_City.groupby("AdvisorContactHomeCity", as_index=False).agg(AdvCount=("AdvCount", "sum"), Submits=("Submits", "sum"),AppCount=("AppCount", "sum"))

out = NC_AdvCount_City1.to_csv (r'C:\Users\test\Documents\SuperCE\NC_AdvCount_City1.csv', index = None, header=True)

## MI

q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvCount, sum(SubmitAmount) as Submits, count(SubmitDate) as AppCount, AdvisorContactHomeStateProvince,AdvisorContactHomeCity FROM SubmittedAdv_2022 where AdvisorContactHomeStateProvince='MI' group by AdvisorContactHomeStateProvince,AdvisorContactHomeCity order by AdvisorContactHomeCity, sum(SubmitAmount) desc ;"""

MI_AdvCount_City =  pysqldf(q3)

MI_AdvCount_City.info()

MI_AdvCount_City['AdvisorContactHomeCity'] = MI_AdvCount_City['AdvisorContactHomeCity'].str.upper()

MI_AdvCount_City1=MI_AdvCount_City.groupby("AdvisorContactHomeCity", as_index=False).agg(AdvCount=("AdvCount", "sum"), Submits=("Submits", "sum"),AppCount=("AppCount", "sum"))

out = MI_AdvCount_City1.to_csv (r'C:\Users\test\Documents\SuperCE\MI_AdvCount_City1.csv', index = None, header=True)

## MA

q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvCount, sum(SubmitAmount) as Submits, count(SubmitDate) as AppCount, AdvisorContactHomeStateProvince,AdvisorContactHomeCity FROM SubmittedAdv_2022 where AdvisorContactHomeStateProvince='MA' group by AdvisorContactHomeStateProvince,AdvisorContactHomeCity order by AdvisorContactHomeCity, sum(SubmitAmount) desc ;"""

MA_AdvCount_City =  pysqldf(q3)

MA_AdvCount_City.info()

MA_AdvCount_City['AdvisorContactHomeCity'] = MA_AdvCount_City['AdvisorContactHomeCity'].str.upper()

MA_AdvCount_City1=MA_AdvCount_City.groupby("AdvisorContactHomeCity", as_index=False).agg(AdvCount=("AdvCount", "sum"), Submits=("Submits", "sum"),AppCount=("AppCount", "sum"))

out = MA_AdvCount_City1.to_csv (r'C:\Users\test\Documents\SuperCE\MA_AdvCount_City1.csv', index = None, header=True)

## IL

q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvCount, sum(SubmitAmount) as Submits, count(SubmitDate) as AppCount, AdvisorContactHomeStateProvince,AdvisorContactHomeCity FROM SubmittedAdv_2022 where AdvisorContactHomeStateProvince='IL' group by AdvisorContactHomeStateProvince,AdvisorContactHomeCity order by AdvisorContactHomeCity, sum(SubmitAmount) desc ;"""

IL_AdvCount_City =  pysqldf(q3)

IL_AdvCount_City.info()

IL_AdvCount_City['AdvisorContactHomeCity'] = IL_AdvCount_City['AdvisorContactHomeCity'].str.upper()

IL_AdvCount_City1=IL_AdvCount_City.groupby("AdvisorContactHomeCity", as_index=False).agg(AdvCount=("AdvCount", "sum"), Submits=("Submits", "sum"),AppCount=("AppCount", "sum"))

out = IL_AdvCount_City1.to_csv (r'C:\Users\test\Documents\SuperCE\IL_AdvCount_City1.csv', index = None, header=True)

## PA

q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvCount, sum(SubmitAmount) as Submits, count(SubmitDate) as AppCount, AdvisorContactHomeStateProvince,AdvisorContactHomeCity FROM SubmittedAdv_2022 where AdvisorContactHomeStateProvince='PA' group by AdvisorContactHomeStateProvince,AdvisorContactHomeCity order by AdvisorContactHomeCity, sum(SubmitAmount) desc ;"""

PA_AdvCount_City =  pysqldf(q3)

PA_AdvCount_City.info()

PA_AdvCount_City['AdvisorContactHomeCity'] = PA_AdvCount_City['AdvisorContactHomeCity'].str.upper()

PA_AdvCount_City1=PA_AdvCount_City.groupby("AdvisorContactHomeCity", as_index=False).agg(AdvCount=("AdvCount", "sum"), Submits=("Submits", "sum"),AppCount=("AppCount", "sum"))

out = PA_AdvCount_City1.to_csv (r'C:\Users\test\Documents\SuperCE\PA_AdvCount_City1.csv', index = None, header=True)

## MD

q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvCount, sum(SubmitAmount) as Submits, count(SubmitDate) as AppCount, AdvisorContactHomeStateProvince,AdvisorContactHomeCity FROM SubmittedAdv_2022 where AdvisorContactHomeStateProvince='MD' group by AdvisorContactHomeStateProvince,AdvisorContactHomeCity order by AdvisorContactHomeCity, sum(SubmitAmount) desc ;"""

MD_AdvCount_City =  pysqldf(q3)

MD_AdvCount_City.info()

MD_AdvCount_City['AdvisorContactHomeCity'] = MD_AdvCount_City['AdvisorContactHomeCity'].str.upper()

MD_AdvCount_City1=MD_AdvCount_City.groupby("AdvisorContactHomeCity", as_index=False).agg(AdvCount=("AdvCount", "sum"), Submits=("Submits", "sum"),AppCount=("AppCount", "sum"))

out = MD_AdvCount_City1.to_csv (r'C:\Users\test\Documents\SuperCE\MD_AdvCount_City1.csv', index = None, header=True)

Writer = pd.ExcelWriter("C:/Users/test/Documents/SuperCE/CombinedState.xlsx", engine="xlsxwriter")

# Write each dataframe to a different worksheet.
FL_AdvCount_City1.to_excel(Writer, sheet_name='FL_2022')
CA_AdvCount_City1.to_excel(Writer, sheet_name='CA_2022')
TX_AdvCount_City1.to_excel(Writer, sheet_name='TX_2022')
AZ_AdvCount_City1.to_excel(Writer, sheet_name='AZ_2022')
NC_AdvCount_City1.to_excel(Writer, sheet_name='NC_2022')
MI_AdvCount_City1.to_excel(Writer, sheet_name='MI_2022')
MA_AdvCount_City1.to_excel(Writer, sheet_name='MA_2022')
IL_AdvCount_City1.to_excel(Writer, sheet_name='IL_2022')
PA_AdvCount_City1.to_excel(Writer, sheet_name='PA_2022')
MD_AdvCount_City1.to_excel(Writer, sheet_name='MD_2022')

# Close the Pandas Excel writer and output the Excel file.
Writer.close()

#### 2019_2023 Jan Data

SubmittedAdv_last4Yr = pd.read_csv('C:/Users/test/Documents/SuperCE/AdvisorSubmit_01012019_09012023.csv',encoding= 'iso-8859-1')

SubmittedAdv_last4Yr.columns = SubmittedAdv_last4Yr.columns.str.replace(' ', '')

SubmittedAdv_last4Yr.info()

q3  = """SELECT count(AdvisorContactIDText), AdvisorContactHomeStateProvince FROM SubmittedAdv_last4Yr group by AdvisorContactHomeStateProvince;"""

State_AdvCount_last4yr =  pysqldf(q3)

###FL

q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvCount, sum(SubmitAmount) as Submits, count(SubmitDate) as AppCount, AdvisorContactHomeStateProvince,AdvisorContactHomeCity FROM SubmittedAdv_last4Yr where AdvisorContactHomeStateProvince='FL' group by AdvisorContactHomeStateProvince,AdvisorContactHomeCity order by AdvisorContactHomeCity, sum(SubmitAmount) desc ;"""

FL_AdvCount_City_4yr =  pysqldf(q3)

FL_AdvCount_City_4yr.info()

FL_AdvCount_City_4yr['AdvisorContactHomeCity'] = FL_AdvCount_City_4yr['AdvisorContactHomeCity'].str.upper()

FL_AdvCount_City1_4yr=FL_AdvCount_City_4yr.groupby("AdvisorContactHomeCity", as_index=False).agg(AdvCount=("AdvCount", "sum"), Submits=("Submits", "sum"),AppCount=("AppCount", "sum"))

out = FL_AdvCount_City1_4yr.to_csv (r'C:\Users\test\Documents\SuperCE\FL_AdvCount_City1_4yr.csv', index = None, header=True)

## CA 
q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvCount, sum(SubmitAmount) as Submits, count(SubmitDate) as AppCount, AdvisorContactHomeStateProvince,AdvisorContactHomeCity FROM SubmittedAdv_last4Yr where AdvisorContactHomeStateProvince='CA' group by AdvisorContactHomeStateProvince,AdvisorContactHomeCity order by AdvisorContactHomeCity, sum(SubmitAmount) desc ;"""

CA_AdvCount_City_4yr =  pysqldf(q3)

CA_AdvCount_City_4yr.info()

CA_AdvCount_City_4yr['AdvisorContactHomeCity'] = CA_AdvCount_City_4yr['AdvisorContactHomeCity'].str.upper()

CA_AdvCount_City1_4yr=CA_AdvCount_City_4yr.groupby("AdvisorContactHomeCity", as_index=False).agg(AdvCount=("AdvCount", "sum"), Submits=("Submits", "sum"),AppCount=("AppCount", "sum"))

out = CA_AdvCount_City1_4yr.to_csv (r'C:\Users\test\Documents\SuperCE\CA_AdvCount_City1_4yr.csv', index = None, header=True)

## TX
q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvCount, sum(SubmitAmount) as Submits, count(SubmitDate) as AppCount, AdvisorContactHomeStateProvince,AdvisorContactHomeCity FROM SubmittedAdv_last4Yr where AdvisorContactHomeStateProvince='TX' group by AdvisorContactHomeStateProvince,AdvisorContactHomeCity order by AdvisorContactHomeCity, sum(SubmitAmount) desc ;"""

TX_AdvCount_City_4yr =  pysqldf(q3)

TX_AdvCount_City_4yr.info()

TX_AdvCount_City_4yr['AdvisorContactHomeCity'] = TX_AdvCount_City_4yr['AdvisorContactHomeCity'].str.upper()

TX_AdvCount_City1_4yr=TX_AdvCount_City_4yr.groupby("AdvisorContactHomeCity", as_index=False).agg(AdvCount=("AdvCount", "sum"), Submits=("Submits", "sum"),AppCount=("AppCount", "sum"))

out = TX_AdvCount_City1_4yr.to_csv (r'C:\Users\test\Documents\SuperCE\TX_AdvCount_City1_4yr.csv', index = None, header=True)

## AZ
q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvCount, sum(SubmitAmount) as Submits, count(SubmitDate) as AppCount, AdvisorContactHomeStateProvince,AdvisorContactHomeCity FROM SubmittedAdv_last4Yr where AdvisorContactHomeStateProvince='AZ' group by AdvisorContactHomeStateProvince,AdvisorContactHomeCity order by AdvisorContactHomeCity, sum(SubmitAmount) desc ;"""

AZ_AdvCount_City_4yr =  pysqldf(q3)

AZ_AdvCount_City_4yr.info()

AZ_AdvCount_City_4yr['AdvisorContactHomeCity'] = AZ_AdvCount_City_4yr['AdvisorContactHomeCity'].str.upper()

AZ_AdvCount_City1_4yr=AZ_AdvCount_City.groupby("AdvisorContactHomeCity", as_index=False).agg(AdvCount=("AdvCount", "sum"), Submits=("Submits", "sum"),AppCount=("AppCount", "sum"))

out = AZ_AdvCount_City1_4yr.to_csv (r'C:\Users\test\Documents\SuperCE\AZ_AdvCount_City1_4yr.csv', index = None, header=True)

## NC

q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvCount, sum(SubmitAmount) as Submits, count(SubmitDate) as AppCount, AdvisorContactHomeStateProvince,AdvisorContactHomeCity FROM SubmittedAdv_last4Yr where AdvisorContactHomeStateProvince='NC' group by AdvisorContactHomeStateProvince,AdvisorContactHomeCity order by AdvisorContactHomeCity, sum(SubmitAmount) desc ;"""

NC_AdvCount_City_4yr =  pysqldf(q3)

NC_AdvCount_City_4yr.info()

NC_AdvCount_City_4yr['AdvisorContactHomeCity'] = NC_AdvCount_City_4yr['AdvisorContactHomeCity'].str.upper()

NC_AdvCount_City1_4yr=NC_AdvCount_City.groupby("AdvisorContactHomeCity", as_index=False).agg(AdvCount=("AdvCount", "sum"), Submits=("Submits", "sum"),AppCount=("AppCount", "sum"))

out = NC_AdvCount_City1_4yr.to_csv (r'C:\Users\test\Documents\SuperCE\NC_AdvCount_City1_4yr.csv', index = None, header=True)


## MI

q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvCount, sum(SubmitAmount) as Submits, count(SubmitDate) as AppCount, AdvisorContactHomeStateProvince,AdvisorContactHomeCity FROM SubmittedAdv_last4Yr where AdvisorContactHomeStateProvince='MI' group by AdvisorContactHomeStateProvince,AdvisorContactHomeCity order by AdvisorContactHomeCity, sum(SubmitAmount) desc ;"""

MI_AdvCount_City_4yr =  pysqldf(q3)

MI_AdvCount_City_4yr.info()

MI_AdvCount_City_4yr['AdvisorContactHomeCity'] = MI_AdvCount_City_4yr['AdvisorContactHomeCity'].str.upper()

MI_AdvCount_City1_4yr=MI_AdvCount_City.groupby("AdvisorContactHomeCity", as_index=False).agg(AdvCount=("AdvCount", "sum"), Submits=("Submits", "sum"),AppCount=("AppCount", "sum"))

out = MI_AdvCount_City1_4yr.to_csv (r'C:\Users\test\Documents\SuperCE\MI_AdvCount_City1_4yr.csv', index = None, header=True)

## MA

q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvCount, sum(SubmitAmount) as Submits, count(SubmitDate) as AppCount, AdvisorContactHomeStateProvince,AdvisorContactHomeCity FROM SubmittedAdv_last4Yr where AdvisorContactHomeStateProvince='MA' group by AdvisorContactHomeStateProvince,AdvisorContactHomeCity order by AdvisorContactHomeCity, sum(SubmitAmount) desc ;"""

MA_AdvCount_City_4yr =  pysqldf(q3)

MA_AdvCount_City_4yr.info()

MA_AdvCount_City_4yr['AdvisorContactHomeCity'] = MA_AdvCount_City_4yr['AdvisorContactHomeCity'].str.upper()

MA_AdvCount_City1_4yr=MA_AdvCount_City.groupby("AdvisorContactHomeCity", as_index=False).agg(AdvCount=("AdvCount", "sum"), Submits=("Submits", "sum"),AppCount=("AppCount", "sum"))

out = MA_AdvCount_City1_4yr.to_csv (r'C:\Users\test\Documents\SuperCE\MA_AdvCount_City1_4yr.csv', index = None, header=True)


## MD

q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvCount, sum(SubmitAmount) as Submits, count(SubmitDate) as AppCount, AdvisorContactHomeStateProvince,AdvisorContactHomeCity FROM SubmittedAdv_last4Yr where AdvisorContactHomeStateProvince='MD' group by AdvisorContactHomeStateProvince,AdvisorContactHomeCity order by AdvisorContactHomeCity, sum(SubmitAmount) desc ;"""

MD_AdvCount_City_4yr =  pysqldf(q3)

MD_AdvCount_City_4yr.info()

MD_AdvCount_City_4yr['AdvisorContactHomeCity'] = MD_AdvCount_City_4yr['AdvisorContactHomeCity'].str.upper()

MD_AdvCount_City1_4yr=MD_AdvCount_City.groupby("AdvisorContactHomeCity", as_index=False).agg(AdvCount=("AdvCount", "sum"), Submits=("Submits", "sum"),AppCount=("AppCount", "sum"))

out = MD_AdvCount_City1_4yr.to_csv (r'C:\Users\test\Documents\SuperCE\MD_AdvCount_City1_4yr.csv', index = None, header=True)


## PA

q3  = """SELECT count(distinct(AdvisorContactIDText)) as AdvCount, sum(SubmitAmount) as Submits, count(SubmitDate) as AppCount, AdvisorContactHomeStateProvince,AdvisorContactHomeCity FROM SubmittedAdv_last4Yr where AdvisorContactHomeStateProvince='PA' group by AdvisorContactHomeStateProvince,AdvisorContactHomeCity order by AdvisorContactHomeCity, sum(SubmitAmount) desc ;"""

PA_AdvCount_City_4yr =  pysqldf(q3)

PA_AdvCount_City_4yr.info()

PA_AdvCount_City_4yr['AdvisorContactHomeCity'] = PA_AdvCount_City_4yr['AdvisorContactHomeCity'].str.upper()

PA_AdvCount_City1_4yr=PA_AdvCount_City.groupby("AdvisorContactHomeCity", as_index=False).agg(AdvCount=("AdvCount", "sum"), Submits=("Submits", "sum"),AppCount=("AppCount", "sum"))

out = PA_AdvCount_City1_4yr.to_csv (r'C:\Users\test\Documents\SuperCE\PA_AdvCount_City1_4yr.csv', index = None, header=True)


Writer1 = pd.ExcelWriter("C:/Users/test/Documents/SuperCE/CombinedState4yr.xlsx", engine="xlsxwriter")

# Write each dataframe to a different worksheet.
FL_AdvCount_City1_4yr.to_excel(Writer1, sheet_name='FL')
CA_AdvCount_City1_4yr.to_excel(Writer1, sheet_name='CA')
TX_AdvCount_City1_4yr.to_excel(Writer1, sheet_name='TX')
AZ_AdvCount_City1_4yr.to_excel(Writer1, sheet_name='AZ')
NC_AdvCount_City1_4yr.to_excel(Writer1, sheet_name='NC')
MI_AdvCount_City1_4yr.to_excel(Writer1, sheet_name='MI')
MA_AdvCount_City1_4yr.to_excel(Writer1, sheet_name='MA')
IL_AdvCount_City1_4yr.to_excel(Writer1, sheet_name='IL')
MD_AdvCount_City1_4yr.to_excel(Writer1, sheet_name='MD')
PA_AdvCount_City1_4yr.to_excel(Writer1, sheet_name='PA')

# Close the Pandas Excel writer and output the Excel file.
Writer1.close()


