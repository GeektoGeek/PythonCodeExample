
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
from pandasql import sqldf
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer   
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
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

## Dataset 1

AnnuitySub3Yrs = pd.read_csv('C:/Users/test/Documents/AnnuitySubmits_3Years/Salesforce/Submit_08012016until08012019.csv',encoding= 'iso-8859-1')

AnnuitySub3Yrs.columns= AnnuitySub3Yrs.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

AnnuitySub3Yrs.info()

### Change the date format

AnnuitySub3Yrs['submit_date'] = pd.to_datetime(AnnuitySub3Yrs['submit_date'])


### Create the database
con = sqlite3.connect("AnnuitySub3Yrs.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
AnnuitySub3Yrs.to_sql("AnnuitySub3Yrs", con, if_exists='replace')

### Producers(Advissors) information: let's cut the data based on 3 separate years first

Ads_Full3Yrs = pd.read_sql("SELECT advisor_name, submit_date, submit_amount FROM AnnuitySub3Yrs ",con)

export_csv = Ads_Full3Yrs.to_csv (r'C:\Users\test\Documents\AnnuitySubmits_3Years\Salesforce\ListGenThroughSQL\Ads_Full3Yrs.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

### August 2016 until July 2017
Ads_Aug2016_July2017 = pd.read_sql("SELECT advisor_name, submit_date, submit_amount FROM AnnuitySub3Yrs where submit_date>= '2016-08-01' and submit_date < '2017-08-01' order by submit_date",con)

export1_csv = Ads_Aug2016_July2017.to_csv (r'C:\Users\test\Documents\AnnuitySubmits_3Years\Salesforce\ListGenThroughSQL\Ads_Aug2016_July2017.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

### August 2017 until July 2018

Ads_Aug2017_July2018_ch = pd.read_sql("SELECT advisor_name, submit_date, submit_amount FROM AnnuitySub3Yrs where submit_date>= '2017-08-01' and submit_date < '2018-08-01' order by submit_date",con)

export3_csv = Ads_Aug2017_July2018_ch.to_csv (r'C:\Users\test\Documents\AnnuitySubmits_3Years\Salesforce\ListGenThroughSQL\Ads_Aug2017_July2018_ch.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

### August 2018 until Aug 1st 2019

Ads_Aug2018_Aug1st2019 = pd.read_sql("SELECT advisor_name, submit_date, submit_amount FROM AnnuitySub3Yrs where submit_date>= '2018-08-01' and submit_date < '2019-08-02' order by submit_date",con)

export2_csv = Ads_Aug2018_Aug1st2019.to_csv (r'C:\Users\test\Documents\AnnuitySubmits_3Years\Salesforce\ListGenThroughSQL\Ads_Aug2018_Aug1st2019 .csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

### Let's try to do a list comparison..

Adv_Aug2018_Aug_2019 = pd.read_csv('C:/Users/test/Documents/AnnuitySubmits_3Years/Salesforce/ListComparisonForSales/ProducersAug2018_Aug2019.csv',encoding= 'iso-8859-1')

### DF1 -is 2018-2019
DFF1 = Adv_Aug2018_Aug_2019.copy()

print(DFF1)


Adv_Aug2017_Aug_2018 = pd.read_csv('C:/Users/test/Documents/AnnuitySubmits_3Years/Salesforce/ListComparisonForSales/ProducersAug2017_Aug2018.csv',encoding= 'iso-8859-1')

### DF2 is 2017-2018
DFF2= Adv_Aug2017_Aug_2018.copy()

print(DFF2)

### Let's look into advisors within Top10%

DFFF1=DFF1[DFF1.CumulativePercentSubmitAmountR <11]

DFFF2=DFF2[DFF2.CumulativePercentSubmitAmountR <11]

print(DFFF1)

print(DFFF2)

##DFFF2_2notin1 = DFFF2[~(DFFF1['AdvisorsName'].isin(DFFF1['AdvisorsName']).dropna().reset_index(drop=True)

### Advisors who were last year did not make it to Aug,2018-Aug,2019 list top 10%

df_delta1_2=DFFF2[DFFF2['AdvisorsName'].apply(lambda x: x not in DFFF1['AdvisorsName'].values)]

print(df_delta1_2)

export2_csv = df_delta.to_csv (r'C:\Users\test\Documents\AnnuitySubmits_3Years\Salesforce\ListComparisonForSales\df_delta.csv', index = None, header=True)

### List of advisors did not exist between Aug,2017-Aug,2018 vs. Aug,2018-Aug,2019

## df_delta1=Adv_Aug2017_Aug_2018[Adv_Aug2017_Aug_2018['AdvisorsName'].apply(lambda x: x not in Adv_Aug2018_Aug_2019['AdvisorsName'].values)]

### print(df_delta1)

### Let's look into advisors within Top11% to Top 20%

DFFF3=DFF1[(DFF1.CumulativePercentSubmitAmountR >= 11) & (DFF1.CumulativePercentSubmitAmountR < 21)]

print(DFFF3)

DFFF4=DFF2[(DFF2.CumulativePercentSubmitAmountR >= 11) & (DFF2.CumulativePercentSubmitAmountR < 21)]

print(DFFF4)

### Advisors who were last year did not make it to Aug,2018-Aug,2019 list between 11% to 20%

df_delta3_4=DFFF4[DFFF4['AdvisorsName'].apply(lambda x: x not in DFFF3['AdvisorsName'].values)]


### Let's look into advisors within Top 21% and Top 30%

DFFF5=DFF1[(DFF1.CumulativePercentSubmitAmountR >= 21) & (DFF1.CumulativePercentSubmitAmountR < 31)]

print(DFFF5)

DFFF6=DFF2[(DFF2.CumulativePercentSubmitAmountR >= 21) & (DFF2.CumulativePercentSubmitAmountR < 31)]

print(DFFF6)

### Advisors who were last year did not make it to Aug,2018-Aug,2019 list between 21% to 30%

df_delta5_6=DFFF6[DFFF6['AdvisorsName'].apply(lambda x: x not in DFFF5['AdvisorsName'].values)]


### Let's look into advisors within Top 31% to Top 40%

DFFF7=DFF1[(DFF1.CumulativePercentSubmitAmountR >= 31) & (DFF1.CumulativePercentSubmitAmountR < 41)]

print(DFFF7)

DFFF8=DFF2[(DFF2.CumulativePercentSubmitAmountR >= 31) & (DFF2.CumulativePercentSubmitAmountR < 41)]

print(DFFF8)

### Advisors who were last year did not make it to Aug,2018-Aug,2019 list between 31% to 40%

df_delta7_8=DFFF8[DFFF8['AdvisorsName'].apply(lambda x: x not in DFFF7['AdvisorsName'].values)]

###

### Let's look into advisors within Top 41% to Top 50%

DFFF9=DFF1[(DFF1.CumulativePercentSubmitAmountR >= 41) & (DFF1.CumulativePercentSubmitAmountR < 51)]

print(DFFF9)

DFFF10=DFF2[(DFF2.CumulativePercentSubmitAmountR >= 41) & (DFF2.CumulativePercentSubmitAmountR < 51)]

print(DFFF10)

### Advisors who were last year did not make it to Aug,2018-Aug,2019 list between 41% to 50%

df_delta9_10=DFFF10[DFFF10['AdvisorsName'].apply(lambda x: x not in DFFF9['AdvisorsName'].values)]


### Let's look into advisors within Top 51% to Top 60%

DFFF11=DFF1[(DFF1.CumulativePercentSubmitAmountR >= 51) & (DFF1.CumulativePercentSubmitAmountR < 61)]

print(DFFF11)

DFFF12=DFF2[(DFF2.CumulativePercentSubmitAmountR >= 51) & (DFF2.CumulativePercentSubmitAmountR < 61)]

print(DFFF12)

### Advisors who were last year did not make it to Aug,2018-Aug,2019 list between 51% to 60%

df_delta11_12=DFFF12[DFFF12['AdvisorsName'].apply(lambda x: x not in DFFF11['AdvisorsName'].values)]


### Let's look into advisors within Top 61% to Top 70%

DFFF13=DFF1[(DFF1.CumulativePercentSubmitAmountR >= 61) & (DFF1.CumulativePercentSubmitAmountR < 71)]

print(DFFF13)

DFFF14=DFF2[(DFF2.CumulativePercentSubmitAmountR >= 61) & (DFF2.CumulativePercentSubmitAmountR < 71)]

print(DFFF14)

### Advisors who were last year did not make it to Aug,2018-Aug,2019 list between 61% to 70%

df_delta13_14=DFFF14[DFFF14['AdvisorsName'].apply(lambda x: x not in DFFF13['AdvisorsName'].values)]


### Let's look into advisors within Top 71% to Top 80%

DFFF15=DFF1[(DFF1.CumulativePercentSubmitAmountR >= 71) & (DFF1.CumulativePercentSubmitAmountR < 81)]

print(DFFF15)

DFFF16=DFF2[(DFF2.CumulativePercentSubmitAmountR >= 71) & (DFF2.CumulativePercentSubmitAmountR < 81)]

print(DFFF16)

### Advisors who were last year did not make it to Aug,2018-Aug,2019 list between 71% to 80%

df_delta15_16=DFFF16[DFFF16['AdvisorsName'].apply(lambda x: x not in DFFF15['AdvisorsName'].values)]

### Let's look into advisors within Top 81% to Top 90%

DFFF17=DFF1[(DFF1.CumulativePercentSubmitAmountR >= 81) & (DFF1.CumulativePercentSubmitAmountR < 91)]

print(DFFF17)

DFFF18=DFF2[(DFF2.CumulativePercentSubmitAmountR >= 81) & (DFF2.CumulativePercentSubmitAmountR < 91)]

print(DFFF18)

### Advisors who were last year did not make it to Aug,2018-Aug,2019 list between 71% to 80%

df_delta17_18=DFFF18[DFFF18['AdvisorsName'].apply(lambda x: x not in DFFF17['AdvisorsName'].values)]


### Let's look into advisors within Top 91% to Top 100%

DFFF19=DFF1[(DFF1.CumulativePercentSubmitAmountR >= 91) & (DFF1.CumulativePercentSubmitAmountR < 101)]

print(DFFF19)

DFFF20=DFF2[(DFF2.CumulativePercentSubmitAmountR >= 91) & (DFF2.CumulativePercentSubmitAmountR < 101)]

print(DFFF20)

### Advisors who were last year did not make it to Aug,2018-Aug,2019 list between 91% to 100%

df_delta19_20=DFFF20[DFFF20['AdvisorsName'].apply(lambda x: x not in DFFF19['AdvisorsName'].values)]


### Let's take a look how to distirbution changes in the top 20% or top 40% level

Top40_18_19=DFF1[(DFF1.CumulativePercentSubmitAmountR > 0) & (DFF1.CumulativePercentSubmitAmountR < 41)]

print(Top40_18_19)

Top40_17_18=DFF2[(DFF2.CumulativePercentSubmitAmountR > 0) & (DFF2.CumulativePercentSubmitAmountR < 41)]

print(Top40_17_18)

### Advisors who were last year did not make it to Aug,2018-Aug,2019 list between 91% to 100%

df_comp=Top40_17_18[Top40_17_18['AdvisorsName'].apply(lambda x: x not in Top40_18_19['AdvisorsName'].values)]










