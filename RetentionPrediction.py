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
import h2o
from h2o.automl import H2OAutoML
h2o.init(max_mem_size='4G')
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

###Marketer Submit Pre 

#MarkPre = pd.read_csv('C:/Users/test/Documents/MarketerBonusCard/MarketerSubmitGrBy_July1_July19th.csv',encoding= 'iso-8859-1')


Submit2019 = pd.read_csv('C:/Users/test/Documents/AdvisorChrun_RetentionModel/DataPrepModel/2019SubmitData.csv',encoding= 'iso-8859-1')

Submit2019.columns = Submit2019.columns.str.replace(' ', '')

Submit2019.columns = Submit2019.columns.str.lstrip()

Submit2019.columns = Submit2019.columns.str.rstrip()

Submit2019.columns = Submit2019.columns.str.strip()

con = sqlite3.connect("Submit2019.db")

Submit2019.to_sql("Submit2019", con, if_exists='replace')

Submit2019.info()


Submit2018 = pd.read_csv('C:/Users/test/Documents/AdvisorChrun_RetentionModel/DataPrepModel/2018SubmitData.csv',encoding= 'iso-8859-1')

Submit2018.columns = Submit2018.columns.str.replace(' ', '')

Submit2018.columns = Submit2018.columns.str.lstrip()

Submit2018.columns = Submit2018.columns.str.rstrip()

Submit2018.columns = Submit2018.columns.str.strip()

con = sqlite3.connect("Submit2018.db")

Submit2018.to_sql("Submit2018", con, if_exists='replace')

Submit2018.info()


Submit2020 = pd.read_csv('C:/Users/test/Documents/AdvisorChrun_RetentionModel/DataPrepModel/2020SubmitData.csv',encoding= 'iso-8859-1')

Submit2020.columns = Submit2020.columns.str.replace(' ', '')

Submit2020.columns = Submit2020.columns.str.lstrip()

Submit2020.columns = Submit2020.columns.str.rstrip()

Submit2020.columns = Submit2020.columns.str.strip()

con = sqlite3.connect("Submit2020.db")

Submit2020.to_sql("Submit2020", con, if_exists='replace')

Submit2020.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, Carrier,ProductCode, MarketerName, OpportunityRiderv1, OpportunityRiderTranslation, AdvisorContactAtheneLastSubmitDate, AdvisorContactNWLastSubmitDate, AdvisorContactAIGFirstSubmitDate01, AdvisorContactAnnuityFirstSubmitdate01   
      FROM Submit2019 group by AdvisorContactIDText;"""
      
Submit2019GrBy =  pysqldf(q3)  

Out4 = Submit2019GrBy.to_csv(r'C:\Users\test\Documents\AdvisorChrun_RetentionModel\DataPrepModel\Submit2019GrBy.csv', index = None, header=True)

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, Carrier,ProductCode, MarketerName, OpportunityRiderv1, OpportunityRiderTranslation, AdvisorContactAtheneLastSubmitDate, AdvisorContactNWLastSubmitDate, AdvisorContactAIGFirstSubmitDate01, AdvisorContactAnnuityFirstSubmitdate01
      FROM Submit2018 group by AdvisorContactIDText;"""
      
Submit2018GrBy =  pysqldf(q3)  

vm1 = """SELECT b.* FROM Submit2019GrBy a INNER JOIN Submit2018GrBy b on ((a.AdvisorContactIDText =b.AdvisorContactIDText));"""      

Common =  pysqldf(vm1)


######

con = sqlite3.connect("Common.db")

Common.to_sql("Common", con, if_exists='replace')

p2  = """SELECT * FROM Submit2018GrBy WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Common);"""
        
FallenAngels2019 =  pysqldf(p2) 

Out4 = FallenAngels2019.to_csv(r'C:\Users\test\Documents\AdvisorChrun_RetentionModel\DataPrepModel\FallenAngels2019.csv', index = None, header=True)

### After you get the Submit2019GrBy and FallenAngels2019 create the RetentionOrNot variable in Excel 1 for Submit2019GrBy and 0 for FallenAngels2019
 
FinalRetentionDataset = pd.read_csv('C:/Users/test/Documents/AdvisorChrun_RetentionModel/DataPrepModel/FinalRetentionDataset.csv',encoding= 'iso-8859-1')

FinalRetentionDataset.info()

Dataframe= FinalRetentionDataset[['AdvisorContactIDText', 'SubmitCnt', 'SubmitAmt', 'Carrier', 'ProductCode', 'OpportunityRiderTranslation','RetainedOrNot']]

Out = Dataframe.to_csv (r'C:\Users\test\Documents\AdvisorChrun_RetentionModel\Dataframe.csv', index = None, header=True)

### Recoding the Carrier variable 

Dataframe = pd.read_csv('C:/Users/test/Documents/AdvisorChrun_RetentionModel/Dataframe.csv',encoding= 'iso-8859-1')


Dataframe['Carrier'] = Dataframe['Carrier'].astype('category')

Dataframe.info()

labels = Dataframe['Carrier'].astype('category').cat.categories.tolist()
replace_map_comp = {'Carrier' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
print(replace_map_comp)
cat_dataframe_replace = Dataframe.copy()
cat_dataframe_replace.replace(replace_map_comp, inplace=True)

### End of Carrier recoding

### Recoding the ProductCode variable 

Dataframe['ProductCode'] = Dataframe['ProductCode'].astype('category')

Dataframe.info()
labels1 = Dataframe['ProductCode'].astype('category').cat.categories.tolist()
replace_map_comp1 = {'ProductCode' : {k: v for k,v in zip(labels1,list(range(1,len(labels1)+1)))}}
print(replace_map_comp1)
cat_dataframe_replace1 = cat_dataframe_replace.copy()
cat_dataframe_replace1.replace(replace_map_comp1, inplace=True)

### Recoding the OpportunityRiderTranslation variable 

Dataframe['OpportunityRiderTranslation'] = Dataframe['OpportunityRiderTranslation'].astype('category')

Dataframe.info()
labels2 = Dataframe['OpportunityRiderTranslation'].astype('category').cat.categories.tolist()
replace_map_comp2 = {'OpportunityRiderTranslation' : {k: v for k,v in zip(labels2,list(range(1,len(labels1)+1)))}}
print(replace_map_comp2)
cat_dataframe_replace1 = cat_dataframe_replace.copy()
cat_dataframe_replace1.replace(replace_map_comp2, inplace=True)

### End of ProductCode recoding

cat_dataframe_replace1['Carrier'] = cat_dataframe_replace1['Carrier'].astype('category')
cat_dataframe_replace1['ProductCode'] = cat_dataframe_replace1['ProductCode'].astype('category')
cat_dataframe_replace1['OpportunityRiderTranslation'] = cat_dataframe_replace1['ProductCode'].astype('category')
cat_dataframe_replace1.info()

Retained_Proportion=cat_dataframe_replace1['RetainedOrNot'].value_counts()/cat_dataframe_replace1['RetainedOrNot'].value_counts().sum()

cat_dataframe_replace1.info()

from sklearn.model_selection import train_test_split

train2, test2 = train_test_split(cat_dataframe_replace1, test_size=0.2)

train2.info()
test2.info()

#load data as h2o frames
train = h2o.H2OFrame(train2)
test = h2o.H2OFrame(test2)

#drop AdvisorContactIDText from data set
AddId = test['AdvisorContactIDText']
train = train.drop('AdvisorContactIDText',axis =1)
test = test.drop('AdvisorContactIDText',axis =1)

#identify predictors and labels
x = train.columns
y = 'RetainedOrNot'
x.remove(y)

#for binary classification, lables should be a factor
train[y] = train[y].asfactor()

#train[y].info()

# Run AutoML
aml = H2OAutoML(max_runtime_secs = 350, max_models= 12, seed= 1,nfolds= 10)
aml.train(x = x, y = y, training_frame = train)
          
#check the leaderboard
lb = aml.leaderboard

lb1= lb.as_data_frame()

lb.head(rows=lb.nrows)

#prediction

preds = aml.leader.predict(test)

 ## Get leaderboard with `extra_columns` = 'ALL'
lb2 = h2o.automl.get_leaderboard(aml, extra_columns = 'ALL')

lb3= lb2.as_data_frame()

#prediction

preds = aml.leader.predict(test)


# Get model ids for all models in the AutoML Leaderboard

model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])

# Get the GBM model
model = h2o.get_model([mid for mid in model_ids if "GBM" in mid][0])  

#metalearner = h2o.get_model(se.metalearner()['name'])

##metalearner = h2o.get_model(aml.leader.metalearner()['name'])
##metalearner.std_coef_plot()

#metalearner.varimp() 


##GBM Prediction

preds_1 = model.predict(test)

model.varimp_plot(num_of_features = 9)

test1=test.cbind(preds_1["predict"])

test1=test1.as_data_frame()

### Lets bring the fullcontatc data

annexus_FullContactDataDump = pd.read_csv('C:/Users/test/Documents/AdvisorChrun_RetentionModel/DataPrepModel/annexus_FullContactDataDump_20210615.csv',encoding= 'iso-8859-1')

annexus_FullContactDataDump.columns = annexus_FullContactDataDump.columns.str.replace(' ', '')

annexus_FullContactDataDump.columns = annexus_FullContactDataDump.columns.str.lstrip()

annexus_FullContactDataDump.columns = annexus_FullContactDataDump.columns.str.rstrip()

annexus_FullContactDataDump.columns = annexus_FullContactDataDump.columns.str.strip()

con = sqlite3.connect("annexus_FullContactDataDump.db")

annexus_FullContactDataDump.to_sql("annexus_FullContactDataDump", con, if_exists='replace')

FinalRetentionDataset.info()

con = sqlite3.connect("FinalRetentionDataset.db")

FinalRetentionDataset.to_sql("FinalRetentionDataset", con, if_exists='replace')

### From the Inner Join the match rate between FullContact and FinalRetention is pretty high #7224 is the matchrate out of 7353 

p2  = """SELECT a.*, b.* FROM FinalRetentionDataset a INNER JOIN annexus_FullContactDataDump b on a.AdvisorContactIDText= b.cust_recordid;"""
        
FinalRetentionDataset_FC =  pysqldf(p2) 

p2  = """SELECT a.*, b.home_value_estimate, b.loan_to_value_estimate, b.discretionary_income_estimate FROM FinalRetentionDataset a LEFT JOIN annexus_FullContactDataDump b on a.AdvisorContactIDText= b.cust_recordid;"""
        
FinalRetentionDataset_FC_IC =  pysqldf(p2) 

###'household_income_estimate', 'net_worth_range', 'bank_cards_holder','home_value_estimate','loan_to_value_estimate','discretionary_income_estimate','financial_debt_range_estimate','cash_value_balance_of_household_estimate'

### Imputation

from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler,IterativeImputer


#### Now Redo the assignment now we have the Mice Imputer working 

Dataframe1= FinalRetentionDataset_FC[['AdvisorContactIDText', 'SubmitCnt', 'SubmitAmt', 'Carrier', 'ProductCode', 'OpportunityRiderTranslation','RetainedOrNot','home_value_estimate','loan_to_value_estimate','discretionary_income_estimate']]

Dataframe1['Carrier'] = Dataframe1['Carrier'].astype('category')

Dataframe1.info()

labels = Dataframe1['Carrier'].astype('category').cat.categories.tolist()
replace_map_comp11 = {'Carrier' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
print(replace_map_comp11)
cat_dataframe_replace11 = Dataframe1.copy()
cat_dataframe_replace11.replace(replace_map_comp11, inplace=True)

### Recoding the OpportunityRiderTranslation variable 

Dataframe1['OpportunityRiderTranslation'] = Dataframe1['OpportunityRiderTranslation'].astype('category')

Dataframe1.info()
labels2 = Dataframe1['OpportunityRiderTranslation'].astype('category').cat.categories.tolist()
replace_map_comp22 = {'OpportunityRiderTranslation' : {k: v for k,v in zip(labels2,list(range(1,len(labels1)+1)))}}
print(replace_map_comp22)
cat_dataframe_replace22 = cat_dataframe_replace11.copy()
cat_dataframe_replace22.replace(replace_map_comp22, inplace=True)

### Recoding the ProductCode variable 

Dataframe1['ProductCode'] = Dataframe1['ProductCode'].astype('category')

Dataframe.info()
labels1 = Dataframe1['ProductCode'].astype('category').cat.categories.tolist()
replace_map_comp33 = {'ProductCode' : {k: v for k,v in zip(labels1,list(range(1,len(labels1)+1)))}}
print(replace_map_comp33)
cat_dataframe_replace33 = cat_dataframe_replace22.copy()
cat_dataframe_replace33.replace(replace_map_comp33, inplace=True)

### Recoding the ProductCode variable 

cat_dataframe_replace1['Carrier'] = cat_dataframe_replace1['Carrier'].astype('category')
cat_dataframe_replace1['ProductCode'] = cat_dataframe_replace1['ProductCode'].astype('category')
cat_dataframe_replace1['OpportunityRiderTranslation'] = cat_dataframe_replace1['ProductCode'].astype('category')
cat_dataframe_replace33.info()

Retained_Proportion1=cat_dataframe_replace33['RetainedOrNot'].value_counts()/cat_dataframe_replace33['RetainedOrNot'].value_counts().sum()

cat_dataframe_replace33.info()

cat_dataframe_replace33 = cat_dataframe_replace33.sort_values('AdvisorContactIDText')
cat_dataframe_replace33['id'] = (cat_dataframe_replace33['AdvisorContactIDText'] != cat_dataframe_replace33['AdvisorContactIDText'].shift()).cumsum()



cat_dataframe_replace44= cat_dataframe_replace33[['id', 'SubmitCnt', 'SubmitAmt', 'Carrier', 'ProductCode', 'OpportunityRiderTranslation','RetainedOrNot','home_value_estimate','loan_to_value_estimate','discretionary_income_estimate']]


training_data, testing_data = train_test_split(cat_dataframe_replace44, test_size=0.25, random_state=25)
### Lets break the data into train and test same proprotion as the label

## Lets use the imputation from fancyimpute

mice_impute = IterativeImputer()

training_datafill = mice_impute.fit_transform(training_data)

test_datafill = mice_impute.fit_transform(testing_data)

### After the trasnform bring back the column header

training_datafill = pd.DataFrame(training_datafill, columns = training_data.columns)

test_datafill = pd.DataFrame(training_datafill, columns = testing_data.columns)

training_datafill.info()
test_datafill.info()

#load data as h2o frames
train111 = h2o.H2OFrame(training_datafill)
test111 = h2o.H2OFrame(test_datafill)

#drop AdvisorContactIDText from data set
AddId1 = test111['id']
train111 = train111.drop('id',axis =1)
test111 = test111.drop('id',axis =1)

#identify predictors and labels
x = train111.columns
y = 'RetainedOrNot'
x.remove(y)

#for binary classification, lables should be a factor
train111[y] = train111[y].asfactor()

#train[y].info()

# Run AutoML
aml1 = H2OAutoML(max_runtime_secs = 350, max_models= 12, seed= 1,nfolds= 10)
aml1.train(x = x, y = y, training_frame = train111)
          
#check the leaderboard
lb111 = aml1.leaderboard

lb1111= lb111.as_data_frame()

lb111.head(rows=lb.nrows)

#prediction

preds111 = aml1.leader.predict(test111)

 ## Get leaderboard with `extra_columns` = 'ALL'
lb222 = h2o.automl.get_leaderboard(aml1, extra_columns = 'ALL')

lb3333= lb222.as_data_frame()

#prediction


# Get model ids for all models in the AutoML Leaderboard

model_ids1 = list(aml1.leaderboard['model_id'].as_data_frame().iloc[:,0])

# Get the GBM model
model_1111 = h2o.get_model([mid for mid in model_ids1 if "GBM" in mid][0])  

#metalearner = h2o.get_model(se.metalearner()['name'])

##metalearner = h2o.get_model(aml.leader.metalearner()['name'])
##metalearner.std_coef_plot()

#metalearner.varimp() 

##GBM Prediction

preds_111 = model_1111.predict(test111)

model_1111.varimp_plot(num_of_features = 9)

test111=test111.cbind(preds_111["predict"])

test111=test111.as_data_frame()
  
h2o.cluster().shutdown()



