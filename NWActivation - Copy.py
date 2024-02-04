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


NationwideSubmits2016_2022= pd.read_csv('C:/Users/test/Documents/Nationwide2022_3YearFAActivation/NationwideSubmits2016_2022.csv',encoding= 'iso-8859-1')

NationwideSubmits2016_2022.columns = NationwideSubmits2016_2022.columns.str.replace(' ', '')

NationwideSubmits2016_2022.info()

NationwideSubmits2016_2022['SubmitDate'] = pd.to_datetime(NationwideSubmits2016_2022['SubmitDate'])

NationwideSubmits2016_2022.groupby(NationwideSubmits2016_2022["SubmitDate"].dt.month).count().plot(kind="bar")

df_2016 = NationwideSubmits2016_2022[(NationwideSubmits2016_2022['SubmitDate'] > "2015-12-31") & (NationwideSubmits2016_2022['SubmitDate'] < "2017-01-01")]

df_2017 = NationwideSubmits2016_2022[(NationwideSubmits2016_2022['SubmitDate'] > "2016-12-31") & (NationwideSubmits2016_2022['SubmitDate'] < "2018-01-01")]

df_2018 = NationwideSubmits2016_2022[(NationwideSubmits2016_2022['SubmitDate'] > "2017-12-31") & (NationwideSubmits2016_2022['SubmitDate'] < "2019-01-01")]

df_2019 = NationwideSubmits2016_2022[(NationwideSubmits2016_2022['SubmitDate'] > "2018-12-31") & (NationwideSubmits2016_2022['SubmitDate'] < "2020-01-01")]

df_2020 = NationwideSubmits2016_2022[(NationwideSubmits2016_2022['SubmitDate'] > "2019-12-31") & (NationwideSubmits2016_2022['SubmitDate'] < "2021-01-01")]

df_2021 = NationwideSubmits2016_2022[(NationwideSubmits2016_2022['SubmitDate'] > "2020-12-31") & (NationwideSubmits2016_2022['SubmitDate'] < "2022-01-01")]

df_2022 = NationwideSubmits2016_2022[(NationwideSubmits2016_2022['SubmitDate'] > "2021-12-31") & (NationwideSubmits2016_2022['SubmitDate'] < "2023-01-01")]

df_2022.info()

result_2016 = df_2016.groupby([df_2016['SubmitDate'].dt.year, df_2016['SubmitDate'].dt.month]).agg({'SubmitAmount':sum})

result_2017 = df_2017.groupby([df_2017['SubmitDate'].dt.year, df_2017['SubmitDate'].dt.month]).agg({'SubmitAmount':sum})

result_2018 = df_2018.groupby([df_2018['SubmitDate'].dt.year, df_2018['SubmitDate'].dt.month]).agg({'SubmitAmount':sum})

result_2019 = df_2019.groupby([df_2019['SubmitDate'].dt.year, df_2019['SubmitDate'].dt.month]).agg({'SubmitAmount':sum})

result_2020 = df_2020.groupby([df_2020['SubmitDate'].dt.year, df_2020['SubmitDate'].dt.month]).agg({'SubmitAmount':sum})

result_2021 = df_2021.groupby([df_2021['SubmitDate'].dt.year, df_2021['SubmitDate'].dt.month]).agg({'SubmitAmount':sum})

result_2022 = df_2022.groupby([df_2022['SubmitDate'].dt.year, df_2022['SubmitDate'].dt.month]).agg({'SubmitAmount':sum})

print(result_2022)

result_2016.reset_index(inplace=True)
result_2016 = result_2016.rename(columns = {'index':'2016MonthYear'})
result_2016['2016SubmitMonthYear']= result_2016['SubmitDate']
result_2016= result_2016[["2016SubmitMonthYear", "SubmitAmount"]]

result_2017.reset_index(inplace=True)
result_2017 = result_2017.rename(columns = {'index':'2017MonthYear'})
result_2017['2017SubmitMonthYear']= result_2017['SubmitDate']
result_2017= result_2017[["2017SubmitMonthYear", "SubmitAmount"]]

result_2018.reset_index(inplace=True)
result_2018 = result_2018.rename(columns = {'index':'2018MonthYear'})
result_2018['2018SubmitMonthYear']= result_2018['SubmitDate']
result_2018= result_2018[["2018SubmitMonthYear", "SubmitAmount"]]

result_2019.reset_index(inplace=True)
result_2019 = result_2019.rename(columns = {'index':'2019MonthYear'})
result_2019['2019SubmitMonthYear']= result_2019['SubmitDate']
result_2019= result_2019[["2019SubmitMonthYear", "SubmitAmount"]]

result_2020.reset_index(inplace=True)
result_2020 = result_2020.rename(columns = {'index':'2020MonthYear'})
result_2020['2020SubmitMonthYear']= result_2020['SubmitDate']
result_2020= result_2020[["2020SubmitMonthYear", "SubmitAmount"]]


result_2021.reset_index(inplace=True)
result_2021 = result_2021.rename(columns = {'index':'2021MonthYear'})
result_2021['2021SubmitMonthYear']= result_2021['SubmitDate']
result_2021= result_2021[["2021SubmitMonthYear", "SubmitAmount"]]


result_2022.reset_index(inplace=True)
result_2022 = result_2022.rename(columns = {'index':'2022MonthYear'})
result_2022['2022SubmitMonthYear']= result_2022['SubmitDate']
result_2022= result_2022[["2022SubmitMonthYear", "SubmitAmount"]]


import plotly.express as px
from plotly.offline import plot

iris= px.data.iris()
fig = px.bar(result_2021, x='2021SubmitMonthYear', y='SubmitAmount')

plot(fig)
#fig = px.bar(result_2021, x="SubmitAmount", y="2021SubmitMonthYear", title="Plot of 2021")
fig.show()

from plotly.subplots import make_subplots

fig = make_subplots(rows=7, cols=1,subplot_titles=("2016 Monthwise Submitted Business", "2017 Monthwise Submitted Business", "2018 Monthwise Submitted Business", "2019 Monthwise Submitted Business","2020 Monthwise Submitted Business","2021 Monthwise Submitted Business","2022 Monthwise Submitted Business"))

fig1 = px.bar(result_2016, x="2016SubmitMonthYear", y="SubmitAmount", title='2016 Results')
fig2 = px.bar(result_2017, x="2017SubmitMonthYear", y="SubmitAmount")
fig3 = px.bar(result_2018, x="2018SubmitMonthYear", y="SubmitAmount")
fig4 = px.bar(result_2019, x="2019SubmitMonthYear", y="SubmitAmount")
fig5 = px.bar(result_2020, x="2020SubmitMonthYear", y="SubmitAmount")
fig6 = px.bar(result_2021, x="2021SubmitMonthYear", y="SubmitAmount")
fig7 = px.bar(result_2022, x="2022SubmitMonthYear", y="SubmitAmount")

fig.add_trace(fig1['data'][0], row=1, col=1)
fig.add_trace(fig2['data'][0], row=2, col=1)
fig.add_trace(fig3['data'][0], row=3, col=1)
fig.add_trace(fig4['data'][0], row=4, col=1)
fig.add_trace(fig5['data'][0], row=5, col=1)
fig.add_trace(fig6['data'][0], row=6, col=1)
fig.add_trace(fig7['data'][0], row=7, col=1)


fig.update_layout(height=800, width=800, title_text="2016-2022 Nationwide Monthly Submitted Business Trends")
fig.show()

plot(fig)


iris= px.data.iris()


out = df_2021.to_csv (r'C:\Users\test\Documents\Nationwide2022_3YearFAActivation\df_2021.csv', index = None, header=True) 

submittotal_2021=df_2021['SubmitAmount'].sum()

con = sqlite3.connect("AtheneAppointed.db")

AtheneAppointed.to_sql("AtheneAppointed", con, if_exists='replace')

AtheneAppointed.info()

AtheneAppointed['AdvisorContactIDText']= AtheneAppointed['ContactID18']

AtheneAppointed['AtheneSalesYTD'] = AtheneAppointed['AnnuitySalesYTD']