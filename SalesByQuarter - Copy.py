

### Make sure to install pandasql 

pip install -U pandasql

### Make sure to install tensorflow
pip install tensorflow

"""
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
import seaborn as sb
from pandasql import sqldf
from sqlalchemy import create_engine
import matplotlib.pyplot as plt


# This line tells the notebook to show plots inside of the notebook

sess = ts.Session()
a = ts.constant(10)
b = ts.constant(32)
print(sess.run(a+b))


data = pd.read_csv('C:/Users/test/Documents/SalesByQuarter/SalesByQuarter.csv')


data_modi = pd.read_csv('C:/Users/test/Documents/SalesByQuarter/SalesByQuarter_modi.csv')
data.head()

data.info()

con = sqlite3.connect("salesbyquarter.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
data.to_sql("salesbyquarter", con, if_exists='replace')

### Run a sql and put it in a dataframe

dfff1 = pd.read_sql("SELECT sum(NumberofTimes2million) as total, LatestMonthwith2MilSales  FROM salesbyquarter WHERE NumberofTimes2million > 20 group by LatestMonthwith2MilSales",con)

dfff2 = pd.read_sql("SELECT sum(NumberofTimes2million) as total, LatestYearwith2MilSales FROM salesbyquarter  WHERE NumberofTimes2million > 20 group by LatestYearwith2MilSales",con)

### Creting the list

### Grouping by FullName and 

dfff3 =pd.read_sql("SELECT count(SFAtheneLastSubmitIDC) as lastsubmittedCount, SFAtheneLastSubmitIDC FROM salesbyquarter group by SFAtheneLastSubmitIDC order by lastsubmittedCount desc",con)
export_csv = dfff3.to_csv (r'C:\Users\test\Documents\SalesByQuarter\SFAtheneLastSubmitIDC.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

dfff4 =pd.read_sql("SELECT count(SFAtheneLastSubmitIDC) as total, SFAtheneLastSubmitIDC, FullName FROM salesbyquarter group by FullName, SFAtheneLastSubmitIDC order by total desc",con)
export_csv = dfff4.to_csv (r'C:\Users\test\Documents\SalesByQuarter\SFAtheneLastSubmitIDC&FullNm.csv', index = None, header=True) ##Don't forget to add '.csv' at the end of the path

dfff5 =pd.read_sql("SELECT count(SFAtheneLastSubmitIDC) as lastsubmittedCount, SFNationwideLastSubmitIDC FROM salesbyquarter group by SFNationwideLastSubmitIDC order by lastsubmittedCount desc",con)
export_csv = dfff5.to_csv (r'C:\Users\test\Documents\SalesByQuarter\SFNationwideLastSubmitIDC.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

dfff6 =pd.read_sql("SELECT count(SFAtheneLastSubmitIDC) as total, SFNationwideLastSubmitIDC, FullName FROM salesbyquarter group by FullName, SFNationwideLastSubmitIDC order by total desc",con)
export_csv = dfff6.to_csv (r'C:\Users\test\Documents\SalesByQuarter\SFNationwideLastSubmitIDC&FullNm.csv', index = None, header=True) ##Don't forget to add '.csv' at the end of the path



dfff1.plot(x='LatestMonthwith2MilSales', y='total', kind='line', 
        figsize=(10, 8), legend=False, style='yo-', label="count 2M Sales total")
plt.axhline(y=total, color='green', linestyle='--', label='count 2M Sales Total')
plt.title("Running The 2M Sales in $X$ LatestMonths\nFrom Same Start Class", y=1.01, fontsize=20)
plt.ylabel("running 2M Sales", labelpad=15)
plt.xlabel("Latest Months", labelpad=15)
plt.legend();

export_csv = dfff1.to_csv (r'C:\Users\test\Documents\SalesByQuarter\dfff1.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


### Ploting a histogram

dfff1['total'].hist()
dfff1.info()

dfff1['total'].hist()

dfff2['total'].hist()

dfff1.plot(x='LatestMonthwith2MilSales', y='total', kind='line')


### Let's create the cursor object

cur = con.cursor()

## Once we have a Cursor object, we can use it to execute a query against the database with the aptly named execute method. The below code will fetch the first 5 rows from the airlines table:

### Test to make sure that the data can be selected from the table
cur.execute("SELECT sum(NumberofTimes2million) as total, LatestYearwith2MilSales  FROM salesbyquarter  WHERE NumberofTimes2million > 20 group by LatestYearwith2MilSal;")

results = cur.fetchall()
print(results)

#dfff=pd.Dataframe(results)


### Looking alll the attributes in the data 

### This is similar to str in R

data1.info()

### Also describe the data

data.describe()

### Use a Seaborn pair Plot

sb.pairplot(data)

# We have to temporarily drop the rows with 'NA' values
# because the Seaborn plotting function does not know
# what to do with them

corr = data.corr()

sns.heatmap(corr)

q1 = """SELECT COUNT(*) FROM data WHERE NumberofTimes2million > 20"""

print(ps.sqldf(q1,locals()))

##q = """SELECT * from data"""

q2 = """SELECT COUNT(NumberofTimes2million) as count, NumberofTimes2million FROM data WHERE NumberofTimes2million > 20"""

print(ps.sqldf(q2,locals()))

q3 = """SELECT NumberofTimes2million, FullName  FROM data WHERE NumberofTimes2million > 20 group by AdvisorKey"""

print(ps.sqldf(q3,locals()))

q4= q3.to_csv()

data.to_csv(index=False)

data_modi.info()

data_modi.plot(x='LatestYearwith2MilSales', y='PercentofNW Sales', kind='line')

q3 = """SELECT NumberofTimes2million, FullName  FROM data WHERE NumberofTimes2million > 20 """

q4 = """SELECT sum(NumberofTimes2million) as total_year, LatestYearwith2MilSales  FROM data_modi WHERE NumberofTimes2million > 20 group by LatestYearwith2MilSales """

print(ps.sqldf(q3,locals()))

print(ps.sqldf(q4,locals()))

q5 = """SELECT sum(NumberofTimes2million) as total, LatestMonthwith2MilSales  FROM data_modi  WHERE NumberofTimes2million > 20 group by LatestMonthwith2MilSales"""

print(ps.sqldf(q5,locals()))

### Put the results into a data frame

q3= pd.DataFrame(q3)

q4= pd.DataFrame(q4)

q5= pd.DataFrame(q5)

cols_to_change = data1['PercentofNW Sales']

for col in cols_to_change:
    data1['PercentofNW Sales'] = data1['PercentofNW Sales'].str.replace('[%,]', '')
    
data1.info()
    
data1.plot(x='LatestMonthwith2MilSales', y='PercentofNW Sales', kind='scatter')

data1['LatestMonthwith2MilSales']=pd.to_numeric(data1['LatestMonthwith2MilSales'])

### Original 

data1['PercentofNW Sales']=pd.to_numeric(data1['PercentofNW Sales'])

    
data1.plot(x='LatestMonthwith2MilSales', y='PercentofNW Sales', kind='scatter')










































