# -*- coding: utf-8 -*-
"""

This program builds the email engagement for the emails

"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This code is written by Dan Sarkar

This is a temporary script file.

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


# This line tells the notebook to show plots inside of the notebook

sess = ts.Session()
a = ts.constant(10)
b = ts.constant(32)
print(sess.run(a+b))


data = pd.read_csv('C:/Users/test/Documents/EmailEngagement/Email_Results_Subscriber_Level_190614a.csv')


data_modi = pd.read_csv('C:/Users/test/Documents/EmailEngagement/Email_Results_Subscriber_Level_190614a.csv')

data_processed= pd.read_csv('C:/Users/test/Documents/EmailEngagement/Email_Results_Subscriber_Level_190614a_processed.csv')


data.head()

data_processed.info()



data.info()

str1= data['Email']

### Let's process the Email here

print(str1)

##Domain= data['Email'].str.split('@', expand=True)

##domain = re.search("@[\w.]+", str)
##print(Domain)

### Note dataframe Domain has two columns separateed by key and values; hence shape is 2

### Pull out the column 1 get the domain

##Domain1=Domain[[1]]

### Append the domain with the appropriate dataset
### data['Domain']=Domain1

sb.pairplot(data)

corr = data.corr()

sns.heatmap(corr)

data.info()

###
data['SendYear'] = pd.DatetimeIndex(data['SendDate']).year

data['SendMonth'] = pd.DatetimeIndex(data['SendDate']).month

data['SendMonth_Year'] = pd.to_datetime(data['SendDate']).dt.to_period('M')

#Let’s check the missing values in the dataset.

data.isnull().sum()

data.describe()

data.info()

print("Schema:\n\n",data.dtypes)
print("Number of questions,columns=",data.shape)

con = sqlite3.connect("EmailEngagement.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
data_modi.to_sql("EmailEngagement", con, if_exists='replace')


data.info()

EmailEngagement_Cat = pd.read_sql("SELECT count(OpenDate) as TotalOpen, count(ClickDate) as TotalClick, Category FROM EmailEngagement group by Category",con)

EmailEngagement_Cat.plot(x='TotalOpen', y='TotalClick', kind='bar')

EmailEngagement_Cat = pd.read_sql("SELECT count(OpenDate) as TotalOpen, count(ClickDate) as TotalClick, Category FROM EmailEngagement group by Category",con)
export_csv = EmailEngagement_Cat.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailEngagement_Cat.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


EmailEngagement_Cat.plot(x='TotalOpen', y='TotalClick', kind='bar')

EmailEngagement_Cat1 = pd.read_sql("SELECT count(SendDate) as TotalSend, count(OpenDate) as TotalOpen, count(ClickDate) as TotalClick, Category FROM EmailEngagement group by Category,MarketingCloudKey,SFDCKey",con)
export_csv = EmailEngagement_Cat1.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailEngagement_Cat1.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

### Let's process the data from SendDate to months
Email = pd.read_sql("SELECT count(SendDate) as TotalSend, count(OpenDate) as TotalOpen, count(ClickDate) as TotalClick, EmailSubject, Category FROM EmailEngagement group by Category,MarketingCloudKey,SFDCKey",con)
export_csv = Email.to_csv (r'C:\Users\test\Documents\EmailEngagement\Email.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

### Let's build a one big query

con = sqlite3.connect("EmailEngagement1.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
data_processed.to_sql("EmailEngagement1", con, if_exists='replace')

EmailEng_all = pd.read_sql("SELECT count(SendDate) as TotalSend, count(OpenDate) as TotalOpen, count(ClickDate) as TotalClick, SendMonth_Year, Category FROM EmailEngagement1 group by Category,SendMonth_Year",con)
export_csv = EmailEng_all.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailEng_all.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path



### Let's look into the total send volume


### Let's Take the data processed Info

con = sqlite3.connect("EmailVolume.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
data_processed.to_sql("EmailVolume", con, if_exists='replace')

data_processed.info()


Send_Volume = pd.read_sql("SELECT count(SendMonth_Year) as TotalSend, SendMonth_Year FROM EmailVolume where SendMonth_Year !='null'  group by SendMonth_Year order by SendMonth_Year desc ",con)
export_csv = Send_Volume.to_csv (r'C:\Users\test\Documents\EmailEngagement\Send_Volume.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

Open_Volume = pd.read_sql("SELECT count(OpenMonth_Year) as TotalOpen, OpenMonth_Year FROM EmailVolume where OpenMonth_Year !='null'  group by OpenMonth_Year order by OpenMonth_Year desc",con)
export_csv = Open_Volume.to_csv (r'C:\Users\test\Documents\EmailEngagement\Open_Volume.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

Click_Volume = pd.read_sql("SELECT count(ClickedMonth_Year) as TotalClicks, ClickedMonth_Year FROM EmailVolume where ClickedMonth_Year !='null'  group by ClickedMonth_Year order by ClickedMonth_Year desc",con)
export_csv = Click_Volume.to_csv (r'C:\Users\test\Documents\EmailEngagement\Click_Volume.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


### Email By Neme Subject line Sends, Opens and Clicks


### By Name
EmailName_DF = pd.read_sql("SELECT count(SendDate) as TotalSend, count(OpenDate) as TotalOpen, count(ClickDate) as TotalClick, Category, EmailName FROM EmailVolume group by EmailName, Category",con)
export_csv = EmailName_DF.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailName_DF.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

EmailSubject_DF = pd.read_sql("SELECT count(SendDate) as TotalSend, count(OpenDate) as TotalOpen, count(ClickDate) as TotalClick, Category, EmailName, EmailSubject FROM EmailVolume group by EmailSubject, EmailName, Category",con)
export_csv = EmailSubject_DF.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailSubject_DF.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

## By Account Name and Branch Member
AccountNm_DF = pd.read_sql("SELECT count(SendDate) as TotalSend, count(OpenDate) as TotalOpen, count(ClickDate) as TotalClick, Category, EmailSubject, Account_Name__c, Branch_Name__c FROM EmailVolume group by EmailSubject,  Category",con)
export_csv = AccountNm_DF.to_csv (r'C:\Users\test\Documents\EmailEngagement\AccountNm_DF.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


### Double Check Slide 3
OpenSendClickCategory_DF = pd.read_sql("SELECT count(SendDate) as TotalSend, count(OpenDate) as TotalOpen, count(ClickDate) as TotalClick, Category, EmailName, EmailSubject FROM EmailVolume group by EmailSubject, EmailName, Category",con)
export_csv = EmailSubject_DF.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailSubject_DF.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path



### Try some NLP


### Subsetiing the DF with specific columns

EmailSubjectNLP= EmailSubject_DF.iloc[:,[0,1,2,5]]


EmailSeqProcessing = pd.read_csv('C:/Users/test/Documents/EmailEngagement/EmailSubjectLineSequentialProcessing.csv')


### Bring the Sequnrtial Email Processed Data


def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("<!--?.*?-->","",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text
 
##EmailSubjectNLP['EmailSubject'] = EmailSubjectNLP['EmailSubject'].apply(lambda x:pre_process(x))

EmailSeqProcessing['EmailSubject'] = EmailSeqProcessing['EmailSubject'].apply(lambda x:pre_process(x))
 
#show the second 'text' just for fun


###Creating Vocabulary and Word Counts for IDF
## Convert the Email Subject to a list
##EmailSubjectNLP['EmailSubject'].to_list()

EmailSeqProcessing['EmailSubject'].to_list()

tfidf = TfidfVectorizer(analyzer='word', stop_words = 'english')

##score = tfidf.fit_transform(EmailSubjectNLP['EmailSubject'])

score = tfidf.fit_transform(EmailSeqProcessing['EmailSubject'])

score1 = tfidf.fit_transform(EmailSeqProcessing['EmailSubject'])
# New data frame containing the tfidf features and their scores
tfidf_df = pd.DataFrame(score.toarray(), columns=tfidf.get_feature_names())

tfidf_df1 = pd.DataFrame(score1.toarray(), columns=tfidf.get_feature_names())

# Filter the tokens with tfidf score greater than 0.3
tokens_above_threshold = tfidf_df.max()[tfidf_df.max() > 0.3].sort_values(ascending=False)

tokens_above_threshold1 = tfidf_df1.max()[tfidf_df1.max() > 0.3].sort_values(ascending=False)

tokens_above_threshold2= pd.DataFrame([tokens_above_threshold1])


tokens_above_threshold2 = tokens_above_threshold2.to_csv (r'C:\Users\test\Documents\EmailEngagement\tokens_above_threshold2.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


### let try to do the WordCloud

EmailSeqProcessing[["EmailSubject"]].head()

country= EmailSeqProcessing['EmailSubject']

# Start with one review:
text = EmailSeqProcessing.EmailSubject[0:17]

wordcloud2 = WordCloud().generate(' '.join(text))

# Create and generate a word cloud image:
## wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
plt.savefig('wordcloud.png', facecolor='k', bbox_inches='tight')



















































