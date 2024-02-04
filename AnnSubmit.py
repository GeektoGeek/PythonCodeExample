
## This is primarily NW Analysis

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
import pingouin as pg
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

AnnuitySubmit = pd.read_csv('C:/Users/test/Documents/AnnuitySubmitData/AnnuitySubmit0101208_06302019.csv',encoding= 'iso-8859-1')


### Cleanup the header

AnnuitySubmit.columns = AnnuitySubmit.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

### Look at the data and run some descriptive stats
AnnuitySubmit.info()

AnnuitySubmit.describe()

## Dataset 2

AnnuityAppAdvisor = pd.read_csv('C:/Users/test/Documents/AnnuitySubmitData/AnnuityAppointmentAdvisorsWithStartDate.csv', encoding= 'iso-8859-1')

AnnuityAppAdvisor.columns= AnnuityAppAdvisor.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

AnnuityAppAdvisor.info()

### Create the database
con = sqlite3.connect("AnnuitySubmit.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
AnnuitySubmit.to_sql("AnnuitySubmit", con, if_exists='replace')


### Create the database
con = sqlite3.connect("AnnuityAppAdvisor.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
AnnuityAppAdvisor.to_sql("AnnuityAppAdvisor", con, if_exists='replace')


### After changing the column name Index to Index1, we are successful

### Check things with Brian Marketer and Associate Marketer and Case Managers

### Lets try to join Submits with their Advisors

### AnnuitySubmit--> advisor_contact_agent_key

###AnnuityAppAdvisor-->agent_key



q  = """SELECT * FROM AnnuitySubmit a
           LEFT JOIN
           AnnuityAppAdvisor b
            ON a.advisor_contact_agent_key = b.agent_key;"""
            
merge_df = ps(q)  

###   Export the data to a csv

export_csv = merge_df.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\Merge_df.csv', index = None, header=True) 

merge_df.info()       

### Now Subsetting the columns


merge_df1= merge_df.iloc[:,[2,5,7,8,9,10,11,12, 20, 21, 22, 23, 24, 25, 26]]

### 26--> agent_key

export_csv = merge_df1.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\Merge_df1.csv', index = None, header=True) 

merge_df1.info()

merge_df1['submit_date'] = pd.to_datetime(merge_df1['submit_date'])



merge_df1['first_nw_adv_appt_start_date2'] = pd.to_datetime(merge_df1['first_nw_adv_appt_start_date2'])

merge_df1['first_athene_adv_appt_start_date2'] = pd.to_datetime(merge_df1['first_athene_adv_appt_start_date2'])

merge_df1['first_ta_adv_appt_start_date2'] = pd.to_datetime(merge_df1['first_ta_adv_appt_start_date2'])

merge_df1.info()

merge_df1['flagNWStartDate'] = merge_df1['first_nw_adv_appt_start_date2'].notnull().astype(int)

merge_df1['flagAtheneStartDate'] = merge_df1['first_athene_adv_appt_start_date2'].notnull().astype(int)

merge_df1['flagTAStartDate'] = merge_df1['first_ta_adv_appt_start_date2'].notnull().astype(int)

merge_df1['flagSubmitDate'] = merge_df1['submit_date'].notnull().astype(int)

con = sqlite3.connect("AnnuitySubmit.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
merge_df1.to_sql("AnnuitySubmit", con, if_exists='replace')

### Lets create four columns to make sure dates have some indicator variable

desc = merge_df1.describe() 

merge_df1.info()

### Now run a query to see the overlaps between start date for theree providers


### Advisors--> Let's review this

advisor_count = pd.read_sql("SELECT advisor_name, submit_date, submit_amount, first_nw_adv_appt_start_date2,first_athene_adv_appt_start_date2, first_ta_adv_appt_start_date2 FROM AnnuitySubmit group by advisor_name",con)

export_csv = advisor_count.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\DataList\advisor_count.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

### Let do another validation of the advsior count using a pivot table without group by

advisor_count1 = pd.read_sql("SELECT advisor_name, submit_date, submit_amount FROM AnnuitySubmit",con)

export1_csv = advisor_count1.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\DataList\advisor_count1.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

### Group By Advisors

### All three provider overlap

advisor_AllthreeOverlap = pd.read_sql("SELECT advisor_name, first_nw_adv_appt_start_date2,first_athene_adv_appt_start_date2, first_ta_adv_appt_start_date2 FROM AnnuitySubmit where flagNWStartDate=1 and flagAtheneStartDate=1 and flagTAStartDate=1 group by advisor_name ",con)

export2_csv = advisor_AllthreeOverlap.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\DataList\advisor_AllthreeOverlap.csv', index = None, header=True)
### All two or three provider overlap

advisor_twoorthreeOverlap = pd.read_sql("SELECT advisor_name, first_nw_adv_appt_start_date2,first_athene_adv_appt_start_date2, first_ta_adv_appt_start_date2 FROM AnnuitySubmit where (flagNWStartDate=1 and flagAtheneStartDate=1 and flagTAStartDate=1) or ((flagNWStartDate=1 and flagAtheneStartDate=1) or (flagNWStartDate=1 and flagTAStartDate=1) or( flagNWStartDate=1 and flagTAStartDate=1)) group by advisor_name ",con)

export3_csv = advisor_twoorthreeOverlap.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\DataList\advisor_twoorthreeOverlap.csv', index = None, header=True)
### All one or two or three provider overlap

advisor_onetwothreeOverlap = pd.read_sql("SELECT advisor_name, first_nw_adv_appt_start_date2,first_athene_adv_appt_start_date2, first_ta_adv_appt_start_date2 FROM AnnuitySubmit where (flagNWStartDate=1 and flagAtheneStartDate=1 and flagTAStartDate=1) or ((flagNWStartDate=1 and flagAtheneStartDate=1) or (flagNWStartDate=1 and flagTAStartDate=1) or( flagNWStartDate=1 and flagTAStartDate=1)) or (flagNWStartDate=1) or (flagAtheneStartDate=1) or (flagTAStartDate=1) group by advisor_name ",con)

export4_csv = advisor_onetwothreeOverlap.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\DataList\advisor_onetwothreeOverlap.csv', index = None, header=True)

## export_csv = Send_Volume.to_csv (r'C:\Users\test\Documents\EmailEngagement\Send_Volume.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

 
### Advisors with product and Submit Amount

advisor_SubmitAmtCount = pd.read_sql("SELECT advisor_name, count(submit_date) as count, sum(submit_amount) as TotalAmt FROM AnnuitySubmit group by advisor_name ",con)


advisor_SubmitAmtCount.info()

advisor_SubmitAmtCount['TotalAmt'] = advisor_SubmitAmtCount['TotalAmt'].astype('int64')

advisor_SubmitAmtCount['TotalAmt'].describe()

corr= advisor_SubmitAmtCount['TotalAmt'].corr(advisor_SubmitAmtCount['count'])


export5_csv = advisor_SubmitAmtCount.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\DataList\advisor_SubmitAmtCount.csv', index = None, header=True) 



g = sns.JointGrid(data=advisor_SubmitAmtCount, x='count', y='TotalAmt', xlim=(1, 485), ylim=(0, 96200000), height=5)
g = g.plot_joint(sns.regplot, color="xkcd:muted blue")
g = g.plot_marginals(sns.distplot, kde=False, bins=12, color="xkcd:bluey grey")
g.ax_joint.text(145, 95, 'r = 0.05, p < .001', fontstyle='italic')
plt.tight_layout()

data2 = advisor_SubmitAmtCount.iloc[:,[1,2]]
corr1 = data2.corr(method="pearson")

corr1

sns.pairplot(corr1)
sns.plt.show()
sns.heatmap(corr1)

merge_df1.info()

### Let's look into some marketer stats
marketer_SubmitAmtCount = pd.read_sql("SELECT marketer_marketer_name, count(submit_date) as count, sum(submit_amount) as TotalAmt FROM AnnuitySubmit group by marketer_marketer_name",con)

export6_csv = marketer_SubmitAmtCount.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\DataList\marketer_SubmitAmtCount.csv', index = None, header=True) 

### Now take this another level further

marketer_SubmitAmtCount = pd.read_sql("SELECT marketer_marketer_name, count(submit_date) as count, sum(submit_amount) as TotalAmt FROM AnnuitySubmit group by marketer_marketer_name",con)

con = sqlite3.connect("marketer_SubmitAmtCount.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
marketer_SubmitAmtCount.to_sql("marketer_SubmitAmtCount", con, if_exists='replace')
## Submit Amount 1-5 and total Submit Amount

mar_one_five= pd.read_sql("SELECT marketer_marketer_name as marketer_name, count, TotalAmt FROM marketer_SubmitAmtCount  where count between 1 and 5 group by marketer_marketer_name",con)

mar_one_five['TotalAmt'] = mar_one_five['TotalAmt'].astype('int64')

export7_csv = mar_one_five.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\DataList\mar_one_five.csv', index = None, header=True) 

mar_six_twenty= pd.read_sql("SELECT marketer_marketer_name as marketer_name, count, TotalAmt FROM marketer_SubmitAmtCount  where count between 6 and 20 group by marketer_marketer_name",con)

mar_six_twenty['TotalAmt'] = mar_six_twenty['TotalAmt'].astype('int64')

export8_csv = mar_six_twenty.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\DataList\mar_six_twenty.csv', index = None, header=True) 

mar_twentyone_fifty= pd.read_sql("SELECT marketer_marketer_name as marketer_name, count, TotalAmt FROM marketer_SubmitAmtCount  where count between 21 and 50 group by marketer_marketer_name",con)

mar_twentyone_fifty['TotalAmt'] = mar_twentyone_fifty['TotalAmt'].astype('int64')

export9_csv = mar_twentyone_fifty.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\DataList\mar_twentyone_fifty.csv', index = None, header=True) 

mar_fiftyone_hundred= pd.read_sql("SELECT marketer_marketer_name as marketer_name, count, TotalAmt FROM marketer_SubmitAmtCount  where count between 51 and 100 group by marketer_marketer_name",con)

mar_fiftyone_hundred['TotalAmt'] = mar_fiftyone_hundred['TotalAmt'].astype('int64')

export10_csv = mar_fiftyone_hundred.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\DataList\mar_fiftyone_hundred.csv', index = None, header=True) 

mar_hundredone_fivehun= pd.read_sql("SELECT marketer_marketer_name as marketer_name, count, TotalAmt FROM marketer_SubmitAmtCount  where count between 101 and 500 group by marketer_marketer_name",con)

mar_hundredone_fivehun['TotalAmt'] = mar_hundredone_fivehun['TotalAmt'].astype('int64')

export10_csv = mar_hundredone_fivehun.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\DataList\mar_hundredone_fivehun.csv', index = None, header=True) 

mar_501_thousand= pd.read_sql("SELECT marketer_marketer_name as marketer_name, count, TotalAmt FROM marketer_SubmitAmtCount  where count between 501 and 1000 group by marketer_marketer_name",con)

mar_501_thousand['TotalAmt'] = mar_501_thousand['TotalAmt'].astype('int64')

export10_csv = mar_501_thousand.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\DataList\mar_501_thousand.csv', index = None, header=True) 

mar_overthousand= pd.read_sql("SELECT marketer_marketer_name as marketer_name, count, TotalAmt FROM marketer_SubmitAmtCount  where count >= 1001 group by marketer_marketer_name",con)

mar_overthousand['TotalAmt'] = mar_overthousand['TotalAmt'].astype('int64')

export10_csv = mar_hundredone_fivehun.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\DataList\mar_overthousand.csv', index = None, header=True) 

### Product Code and Submit Count Breakdown by marketers

marketer_SubmitCount = pd.read_sql("SELECT marketer_marketer_name, count(submit_date) as Submitcount, sum(submit_amount) as TotalAmt, carrier, product_code FROM AnnuitySubmit group by marketer_marketer_name",con)

export11_csv = marketer_SubmitCount.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\DataList\marketer_SubmitCount.csv', index = None, header=True) 

### Let's validate what count distinct brings in the response

AAA = pd.read_sql("SELECT count(distinct(marketer_marketer_name)) as Count_marketer FROM AnnuitySubmit",con)

AAA

### It comes back as 310

marketer_carrier = pd.read_sql("SELECT count(distinct(marketer_marketer_name)), carrier FROM AnnuitySubmit group by carrier",con)

export11_csv = marketer_carrier.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\DataList\marketer_carrier.csv', index = None, header=True) 

marketer_product = pd.read_sql("SELECT count(distinct(marketer_marketer_name)), product_code FROM AnnuitySubmit group by product_code",con)

export11_csv = marketer_product.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\DataList\marketer_product.csv', index = None, header=True) 

### Marketer Advisor Relationship


marketer_AdvisorCount = pd.read_sql("SELECT marketer_marketer_name as marketer_name, count(advisor_name) as numberofAdvisor, sum(submit_amount) as TotalAmt, carrier, product_code FROM AnnuitySubmit group by marketer_marketer_name",con)

marketer_AdvisorCount['TotalAmt'] = marketer_AdvisorCount['TotalAmt'].astype('int64')

export12_csv = marketer_AdvisorCount.to_csv (r'C:\Users\test\Documents\AnnuitySubmitData\DataList\marketer_AdvisorCount.csv', index = None, header=True) 

corr2= marketer_SubmitCount['Submitcount'].corr(marketer_SubmitCount['TotalAmt'])










