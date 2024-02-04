
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as ss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score,precision_score
from sklearn.utils import resample
import itertools

## Part 1: Reading data from text file which is in the json format


with open('transactions.txt') as file:
    status = []
    for line in file:
        status.append(json.loads(line))
        
status[0]        
        
transaction=pd.DataFrame(status)        

transaction.shape

len(transaction)

## These are all the information we have for every transaction

## Lets have a look at the dataframe which has the transactions. Before that lets change max number of columns to display all the columns present in the dataframe.

len(transaction.columns)

transaction.columns

pd.set_option('display.max_columns', None)

transaction.head()

##Lets have a look at number of values which are not present

transaction.isnull().sum(axis=0)

### Above results show that no column has the missing values. However when we had displayed 5 rows in the dataframe, 
###we could see that some of the columns like merchantZip dont have any value. The missing values dont have any 
### datatype and lets replace it by NaN (Not a number which denotes missing value)

transaction=(transaction.replace(r'^\s*$', np.nan, regex=True))

transaction.isnull().sum(axis=0)

### Now we can see that some of the columns have missing values. Lets deal with them

### Some columns have same number of missing values as that of the number of rows (641914). We cant impute those values and lets 
### entirely remove these columns


transaction=transaction.drop(['merchantCity','echoBuffer','merchantState','merchantZip','posOnPremises','recurringAuthInd'],axis=1)

### Lets deal with other columns having missing values

## 1) acqCountry

transaction['acqCountry'].value_counts()/len(transaction)

### Almost 99% of the values in the column has value 'US'. The number of missing values are 3193 which is greater than other values. 
## In this case we cant even replace the values by mean as it is a categorical variable. Now, lets fill the missing values with the 
## element having maximum frequency which is 'US'. We can also use some method like linear regression to predict the values in this case
## but the category having maximum frequency is dominating the column by large margin and so it would have become unbalanced dataset 
## which would have resulted in 'US' only in most of the cases. So we replaced it with the maximum frequency element.

transaction['acqCountry']= transaction['acqCountry'].fillna(transaction['acqCountry'].value_counts().idxmax())

### 2) Transaction Type:

transaction['transactionType'].value_counts()

transaction['transactionType']= transaction['transactionType'].fillna(transaction['transactionType'].value_counts().idxmax())

## 3) Merchant country code

transaction['merchantCountryCode'].value_counts()

transaction['merchantCountryCode']= transaction['merchantCountryCode'].fillna(transaction['merchantCountryCode'].value_counts().idxmax())

## 4) posConditionCode

transaction['posConditionCode'].value_counts()

transaction['posConditionCode']= transaction['posConditionCode'].fillna(transaction['posConditionCode'].value_counts().idxmax())

### 5) posEntryMode

transaction['posEntryMode'].value_counts()

transaction['posEntryMode']= transaction['posEntryMode'].fillna('00')

### In this case, all the categories have almost equal of comparable distribution. So, we cant directly replace it with maximum element. However we can apply multi-class classification to predict the category which will be computationally heavy task for 0.5% of rows for a single column. Also this method doesnt guarantee the correct imputation. So, the simplest thing we can do is we can create another category for the missing values.

### No method guarantees the perfect imputation as it is the complete unsupervised method. We are not aware of the 
## results and can be completely random. Also its not even worth to invest large amount of time and resources to impute very small
## fraction of values.

transaction.isnull().sum(axis=0)

## Now number of missing values are 0 in every column and we have successfully imputed the missing values.

transaction.dtypes

### After having look at the the columns, I realized that some of the columns which are in the numeric format have data types object.
##  Also the columns which contain dates are in the object format. Lets, change the datatypes of these columns


transaction.head(3)

## Converting numeric columns to Int


transaction['accountNumber']=transaction['accountNumber'].astype(str).astype(int)

transaction['cardCVV']=transaction['cardCVV'].astype(str).astype(int)

transaction['cardLast4Digits']=transaction['cardLast4Digits'].astype(str).astype(int)

transaction['enteredCVV']=transaction['enteredCVV'].astype(str).astype(int)

### Converting columns having dates to type datatime


transaction['transactionDateTime'] =  pd.to_datetime(transaction['transactionDateTime'])

transaction['accountOpenDate'] =  pd.to_datetime(transaction['accountOpenDate'])

transaction['currentExpDate'] =  pd.to_datetime(transaction['currentExpDate'])

transaction['dateOfLastAddressChange'] =  pd.to_datetime(transaction['dateOfLastAddressChange'])

display(transaction['transactionAmount'].describe())

display(transaction['creditLimit'].describe())

### Part 2
### Plotting histogram of transactionAmount column with 100 bins

plt.hist(transaction['transactionAmount'],bins=100)
plt.title('histogram of transaction amount')
plt.xlabel('transaction amount')
plt.ylabel('frequency')
plt.show()

### The above diagram shows histogram of all the transactions. Y-axis denotes the frequency of the transaction amount and
### transaction amount increases along the x-axis. We can observe the trend that as amount increases, the number of transactions
### are exponentially decreasing. So, users are using credit cards for lower amount of transactions more often than the larger amount.
### Probably the credit limit is putting restriction on the transactions.

### Lets have a look at transaction amount distribution more closely by dividing it in the different ranges.

plt.hist(transaction[transaction['transactionAmount']<200]['transactionAmount'],bins=100)
plt.title('histogram of transaction amount<200')
plt.xlabel('transaction amount')
plt.ylabel('frequency')
plt.show()

plt.hist(transaction[(transaction['transactionAmount']>=200)&(transaction['transactionAmount']<400)]['transactionAmount'],bins=100)
plt.title('histogram of transaction amount between 200 and 400')
plt.xlabel('transaction amount')
plt.ylabel('frequency')
plt.show()

plt.hist(transaction[(transaction['transactionAmount']>=400)&(transaction['transactionAmount']<600)]['transactionAmount'],bins=100)
plt.title('histogram of transaction amount between 400 and 600')
plt.xlabel('transaction amount')
plt.ylabel('frequency')
plt.show()

plt.hist(transaction[transaction['transactionAmount']>=600]['transactionAmount'],bins=100)
plt.title('histogram of transaction amount more than 600')
plt.xlabel('transaction amount')
plt.ylabel('frequency')
plt.show()

plt.hist(transaction[transaction['transactionAmount']>=1500]['transactionAmount'],bins=100)
plt.title('histogram of transaction amount more than 1500')
plt.xlabel('transaction amount')
plt.ylabel('frequency')
plt.show()


### From the above diagrams we can conclude that almost similar trend is observed for all the ranges. 
### As we increase the amount, the frequency also decreases which can be observed from the y-axis range. 
### For the amount more than 1500, there are very rare transactions and clearly indicates that people dont use credit cards 
### for amount more than 1000 very often and most of the transactions are there for amount less than 500.

transaction['transactionAmount'].mean()

catmean=transaction.groupby('merchantCategoryCode')['transactionAmount'].mean()
catmean

ax = catmean.plot(kind = 'bar',figsize=(12,8))
ax.set_ylabel('Mean Transaction amount for every category')
plt.show()

Fmean=transaction.groupby('isFraud')['transactionAmount'].mean()
ax = Fmean.plot(kind = 'bar',figsize=(12,8),color='R')
ax.set_ylabel('Mean Transaction amount for fraud and non-fraud transactions')
plt.show()

### Average amount for fraud transactions is around 230-240 and for not fraud transaction is much lesser which is 140.
### So, it is evident that fraud transactions are happening for higher amount while normal transactions have lesser average.

Fmean


transaction.groupby('merchantCategoryCode').size()

(transaction.groupby(['accountNumber'])['creditLimit'].mean()).mean()


### Part 3
### To observe the patterns of the transactions, lets display more rows.

pd.options.display.max_rows = 200

transaction.head(500)


### 1) Reversal: Reversal are easier to detect as it is given by transaction type. 
### In case of reversal the amount is credited back to the user account and so available amount increases. 
### There are 16262 reversed transaction. Here are some observations for reversal transactions:

rev=transaction[transaction['transactionType']=='REVERSAL']

print('Number of reversed transactions:',len(rev))
print('Percentage of reversed transactions: ',len(rev)/len(transaction)*100,'%')

print('Total amount of reversed transactions:',sum(rev['transactionAmount']))
print('Fraction of reversed transactions: ',sum(rev['transactionAmount'])/sum(transaction['transactionAmount']))

### 2) Multi-swipe duplicated transaction: 

### These kind of transactions happen due to multiple swipe or multiple payment for the same transaction. 
###In this case, the multiple transactions of same amount happen for the same merchant within short period of time. 
### In the problem statement, the short period of time is not clearly defined and I will assume that multi-swipe 
### transaction reflects on the account within 24 hours of the original amount. (By intuition we know that the transaction happens 
### immediately in most of the cases. But, I am taking longer period as system might take more time to verify some of the transactions 
## and I dont want to rule out such possibilities)

transaction['duplicated'] = ((transaction['transactionType']!='REVERSAL')&
                            (transaction['merchantCategoryCode'] == transaction['merchantCategoryCode'].shift(1))&
                            (transaction['merchantName'] == transaction['merchantName'].shift(1)) & 
                            ((transaction['transactionAmount'] == transaction['transactionAmount'].shift(1)) & 
                            (transaction['accountNumber'] == transaction['accountNumber'])) &
                            (((transaction['transactionDateTime']-transaction['transactionDateTime'].shift(1))/np.timedelta64(1,'D'))<=1))


### This is how the multi-swipe duplicated transactions are detected. The duplicated transaction is the next to its original 
### transaction which happen withing a day. Also, merchant name, amount, are same. Most of the cases are covered by these conditions.
###  It is very well possible that the same item can be purchased from the same merchant on the same day by the same user. But for 
### simplicity I am ignoring this case. If we have to consider this case, we need to reduce the time between these transactions to a minute or
### an hour. But it can also ignore some system changes.
                            
### In case of reversal transactions, same conditions can be satisfied so I am excluding reversal transactions by using first condition.
                            
transaction.head(5)

dupl=transaction[transaction['duplicated']==True]

dupl


print('Number of duplicate multi-swipe transactions:',len(dupl))
print('Percentage of duplicate multi-swipe transactions: ',len(dupl)/len(transaction)*100,'%')

print('Total amount of duplicate multi-swipe transactions:',sum(dupl['transactionAmount']))
print('Fraction of duplicate multi-swipe transactions: ',sum(dupl['transactionAmount'])/sum(transaction['transactionAmount']))

### Part 4

### In this segment, we need to build a model which will successfully detect the fraud transactions. 
### We cant afford to have false negatives in this case. So I will try to build model with high recall. 
### High recall intuitively denotes that the classification model has ability to detect larger extent of positive examples.   


(transaction[transaction['transactionType']=='ADDRESS_VERIFICATION']).groupby('isFraud').size()

(transaction[transaction['transactionType']=='REVERSAL']).groupby('isFraud').size()

### Integer columns
### Finding pearson's correlation coefficient for all the integer variables 

### Its value ranges from -1 to +1 where, -1 denotes negative correlation between the features and +1 denotes very strong 
### positive correlation. 0 denotes no correlation. We need to select the features which has very high positive and negative 
### correlation coefficient with the ‘isfraud’ variable.

plt.figure(figsize=(10,10))
cor = transaction.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

cor_target = abs(cor["isFraud"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.008]
relevant_features

### These are the only features which we can select as only these features have coefficient larger than 0.008. 
### I decided the threshold by observing the values for all the features. We wanted to ignore the insignificant 
### features like accountNumber and CVV which are unique identifies and doesn’t affect transactions by any means.


cor_target[cor_target<0.008]

## Categorical columns
### We can’t find the correlation between categorical and nominal variable by the same method. (Boolean variables are nominal as these variables are numeric but not continuous, as these variables have levels.) 


### Cramer’s V test is the best method to find correlation between categorical or nominal variables. It is based on pearson’s 
## chi squared test. Its value is between 0 and 1 where 0 denotes no correlation and 1 denotes high correlation.

## https://medium.com/@outside2SDs/an-overview-of-correlation-measures-between-categorical-and-continuous-variables-4c7f85610365

def cramers_corrected_stat(confusion_matrix):
    
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

## https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V 
## https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
    

catCols=['isFraud','acqCountry','merchantCategoryCode','merchantCountryCode','merchantName','transactionType','posEntryMode','posConditionCode']

corrM = np.zeros((len(catCols),len(catCols)))
for col1, col2 in itertools.combinations(catCols, 2):
    idx1, idx2 = catCols.index(col1), catCols.index(col2)
    corrM[idx1, idx2] = cramers_corrected_stat(pd.crosstab(transaction[col1], transaction[col2]))
    corrM[idx2, idx1] = corrM[idx1, idx2]
    

corrCat = pd.DataFrame(corrM, index=catCols, columns=catCols)
fig, ax = plt.subplots(figsize=(7, 6))
ax = sns.heatmap(corrCat, annot=True, ax=ax); 
ax.set_title("Cramer V Correlation between Variables");
plt.show()



cor_target = (corrCat["isFraud"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>=0.008]
relevant_features

## We will apply the same threshold which we applied for integer correlations.

#Insignificant features
cor_target[cor_target<0.008]

len(transaction['merchantName'].unique())

print('Number of unique categories',len(transaction['merchantCategoryCode'].unique()))
transaction['merchantCategoryCode'].value_counts()

### There are multiple methods which we can consider for vectorization. We can have a number for every category. (Ordinal encoding).
### However, the algorithm will assume that these are the levels. For example, if online_retail =1 and fastfood = 2, the algorithm will assume that fastfood has more weight than online_retails and so on. However, these are independent categories and there isnt significance leveling between these categories. So using ordinal encoding isnt a good idea.
### We can use 1-hot encoding. In this case, 19 different columns will be created. Each will have value 0 or 1. where, 1 denotes value
## present.
### merchant category denotes the class and from the heatmap we can observe that these two features are highly significant.
### So we can use merchant category only and dropping merchant name wont make significant difference as the same information is
## used in the other column.

transaction['transactionType'].unique()


transaction['posEntryMode'].unique()


transaction['posConditionCode'].unique()

## With transactionType, posEntryMode and posConditionlCode also, same logic (1-hot encoding) can be used for categorical encoding.

# Dropping all the columns which are insignificant

data=transaction.drop(['acqCountry','merchantCountryCode','merchantName','accountNumber','availableMoney','cardCVV','cardLast4Digits','creditLimit','enteredCVV','expirationDateKeyInMatch','duplicated','customerId'],axis=1)

### Creating 1-hot encoded columns for the above descibed features

newData=pd.get_dummies(data, columns=['transactionType','merchantCategoryCode','posEntryMode','posConditionCode'], drop_first=False)

## Date columns
### We can segment this feature to get more information like day of , month, year, day of week, hour in the day. 
### So that we get 5 different features.


newData['day_of_trans']=newData['transactionDateTime'].dt.day
newData['month_of_trans']=newData['transactionDateTime'].dt.month
newData['year_of_trans']=newData['transactionDateTime'].dt.year
newData['hour_of_trans']=newData['transactionDateTime'].dt.hour
newData['week_of_trans']=newData['transactionDateTime'].dt.weekday_name

### We can get current age of account from this feature by subtracting transactionDate from this feature.

newData['age_of_acnt']=(newData['transactionDateTime']-newData['accountOpenDate'])/ np.timedelta64(1, 'D')

### This column denotes the expiration date of the account.
## We can calculate remaining age of the account at the current time using this feature.

newData['age_rem']=(newData['currentExpDate']-newData['transactionDateTime'])/ np.timedelta64(1, 'D')

newData['addr_change']=(newData['transactionDateTime']-newData['dateOfLastAddressChange'])/ np.timedelta64(1, 'D')

## We can again find how long the address was last changed.


newData=newData.drop(['transactionDateTime','currentExpDate','dateOfLastAddressChange','accountOpenDate'],axis=1)

newData=pd.get_dummies(newData, columns=['week_of_trans'], drop_first=False)


newData.dtypes

## As this is fraud detection algorithm and odds of having fraud transactions are very less, we have highly unbalanced dataset.

actCount=newData.groupby('isFraud').size()
actCount/len(newData)*100


# Bar plot of class distribution
ax = actCount.plot(kind = 'bar',figsize=(12,8),title='Distribution of transactions for actual data')
plt.show()

target=newData['isFraud']
finaldf=newData.drop(['isFraud'],axis=1)

### Lets try random forest on the unbalanced data

train_features, test_features, train_labels, test_labels = train_test_split(finaldf, target, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

clf = RandomForestClassifier()

clf.fit(train_features, train_labels)

preds=clf.predict(test_features)

preds_train=clf.predict(train_features)

accuracy_score(preds_train,train_labels)*100

accuracy_score(preds,test_labels)*100

pd.crosstab(test_labels, preds, rownames=['Actual Species'], colnames=['Predicted Species'])

pd.crosstab(train_labels, preds_train, rownames=['Actual Species'], colnames=['Predicted Species'])

precision_score(test_labels,preds)

recall_score(test_labels,preds)

precision_score(test_labels,preds)


recall_score(test_labels,preds)

### Only 13 rows where predicted fraud by the random forest algorithm. There are 2878 true negatives which is not useful in the business sense. So accuracy score is not necessarily useful for imbalanced dataset. Recall score is 0.002 which is really bad to detect actual fraud transactions.

### So we will have to modify the dataset. There are multiple methods to handle unbalanced data.


### Downsampling¶
### In the down sampling approach, dataset is balanced by selecting the same number of majority class as that of minority class.
### So, we will select all the Fraud transactions and the same number of non-fraud transactions. We will build the classification model on the selected data.

newData = newData.sample(frac=1)


frauds = newData.loc[newData['isFraud'] == True]
non_frauds = newData.loc[newData['isFraud'] == False]

underSamp = pd.concat([frauds, non_frauds[:len(frauds)]])

# Shuffle dataframe rows
underSamp = underSamp.sample(frac=1)

underSamp.head()

len(underSamp)

undCount=underSamp.groupby('isFraud').size()
undCount

ax = undCount.plot(kind = 'bar',figsize=(12,8),title='Distribution of transactions when downsampled')
plt.show()

target=underSamp['isFraud']

finalDf=underSamp.drop(['isFraud'],axis=1)

finalDf.head(3)

finalDf.shape

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(finalDf, target, test_size = 0.25, random_state = 42)

### Splitted the dataframe in 75:25. 75% of the data is used to train the classification model and 25% is used to test the model.

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


## Random Forest
### Hyperparameters for tuning to train random forest

param_rf = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

rf = RandomForestClassifier()
grid_search_rf = GridSearchCV(rf, param_grid=param_rf, cv=5, iid=False)
grid_search_rf.fit(train_features, train_labels)

preds=grid_search_rf.predict(test_features)

pd.crosstab(test_labels, preds, rownames=['Actual Species'], colnames=['Predicted Species'])

print('Performance of random forest with undersampling and grid search on test set\n')
print('Precision score: ',precision_score(test_labels, preds, average='macro'))
print('Recall score: ',recall_score(test_labels, preds, average='macro'))
print('Accuracy Score: ',accuracy_score(test_labels, preds))
print('F-1 Score: ',f1_score(test_labels, preds, average='macro'))

preds_train=grid_search_rf.predict(train_features)

pd.crosstab(train_labels, preds_train, rownames=['Actual Species'], colnames=['Predicted Species'])

print('Performance of random forest with undersampling and grid search on train set\n')
print('Precision score: ',precision_score(train_labels, preds_train, average='macro'))
print('Recall score: ',recall_score(train_labels, preds_train, average='macro'))
print('Accuracy Score: ',accuracy_score(train_labels, preds_train))
print('F-1 Score: ',f1_score(train_labels, preds_train, average='macro'))

### We can observe that random forest with undersampling is performing much better than the previous approach. 
### Recall and precision scores are 67%. However, detecting only 2 out of 3 frauds in not good in the fraud transaction detection and we can say that RF is performing poorly with the undersampled data. It has very low test as well as train error. It has high bias and it is not catching all the patterns in the dataset.

### Now lets move to another popular algorithm which is logistic regression.

### Logistic regression

# The hyper parameters are selected from these

lr_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_lr = GridSearchCV(LogisticRegression(), lr_params,cv=5)

grid_lr.fit(train_features, train_labels)

preds_lr=grid_lr.predict(test_features)

pd.crosstab(test_labels, preds_lr, rownames=['Actual Species'], colnames=['Predicted Species'])

print('Performance of Logistic regression with undersampling and grid search on test set\n')
print('Precision score: ',precision_score(test_labels, preds_lr, average='macro'))
print('Recall score: ',recall_score(test_labels, preds_lr, average='macro'))
print('Accuracy Score: ',accuracy_score(test_labels, preds_lr))
print('F-1 Score: ',f1_score(test_labels, preds_lr, average='macro'))


preds_train_lr=grid_lr.predict(train_features)

pd.crosstab(train_labels, preds_train_lr, rownames=['Actual Species'], colnames=['Predicted Species'])

print('Performance of logistic regression with undersampling and grid search on train set\n')
print('Precision score: ',precision_score(train_labels, preds_train_lr, average='macro'))
print('Recall score: ',recall_score(train_labels, preds_train_lr, average='macro'))
print('Accuracy Score: ',accuracy_score(train_labels, preds_train_lr))
print('F-1 Score: ',f1_score(train_labels, preds_train_lr, average='macro'))

###These are the results which we are getting on logistic regression. Even this algorithm is performing poorly with the undersampled data. It also has high bias.

## The problem with undersampled data is, it removes more than 95% of data. Training is done on only 3% of available data.
## So, we might miss important patterns. Also non-fraud data is randomly selected. So the error is subject to change based on the selected data.

### Upsampling
### In this case, we synthetically create the minority class and make it equal to the majority class without removing 
### any of the majority class examples that we did for undersampling. Advantage of this method is we are not lossing any information 
## from the dataset and 100% of data can be used.

# upsample minority
fraud_upsampled = resample(frauds,
                          replace=True, # sample with replacement
                          n_samples=len(non_frauds)) # match number in majority class) 
upsamp = pd.concat([non_frauds, fraud_upsampled])

len(upsamp)

upCount=upsamp.groupby('isFraud').size()
upCount


ax = upCount.plot(kind = 'bar',figsize=(12,8),title='Distribution of transactions when upsampled')
plt.show()


targetup=upsamp['isFraud']

finalDfup=upsamp.drop(['isFraud'],axis=1)

finalDfup.head(3)

finalDfup.shape

# Split the data into training and testing sets
trainup_features, testup_features, trainup_labels, testup_labels = train_test_split(finalDfup, targetup, test_size = 0.25, random_state = 42)

print('Training Features Shape:', trainup_features.shape)
print('Training Labels Shape:', trainup_labels.shape)
print('Testing Features Shape:', testup_features.shape)
print('Testing Labels Shape:', testup_labels.shape)

## Random Forest

rfup=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

rfup.fit(trainup_features, trainup_labels)

predsup=rfup.predict(testup_features)

pd.crosstab(testup_labels, predsup, rownames=['Actual Species'], colnames=['Predicted Species'])

print('Performance of random forest with oversampling and grid search on test set\n')
print('Precision score: ',precision_score(testup_labels, predsup, average='macro'))
print('Recall score: ',recall_score(testup_labels, predsup, average='macro'))
print('Accuracy Score: ',accuracy_score(testup_labels, predsup))
print('F-1 Score: ',f1_score(testup_labels, predsup, average='macro'))

preds_trainup=rfup.predict(trainup_features)

pd.crosstab(trainup_labels, preds_trainup, rownames=['Actual Species'], colnames=['Predicted Species'])

print('Performance of random forest with oversampling and grid search on train set\n')
print('Precision score: ',precision_score(trainup_labels, preds_trainup, average='macro'))
print('Recall score: ',recall_score(trainup_labels, preds_trainup, average='macro'))
print('Accuracy Score: ',accuracy_score(trainup_labels, preds_trainup))
print('F-1 Score: ',f1_score(trainup_labels, preds_trainup, average='macro'))


## We can observe here that random forest is performing very excellent on the oversampled data. For the train set out of 937K examples, only 1 is classified in the wrong class, giving more than 99.99% accuracy. Most importantly, it is not giving a single true negative. It is correctly detecting all the frauds.

## With the test set also, no true negatives are predicted. Accuracy, precision and recall score percentage is more than 99.99.

## So, oversampled data works the best with random forest with hyperparameter tuning. 
###It is detecting every fraud correctly and very negligible false positives which can be tollerated in the fraud transaction detection where detecting fraud is more important than not detecting a fraud transaction.


    

                         





