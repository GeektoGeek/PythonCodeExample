
import pandas as pd
import sqlite3
import numpy as np
from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps


train = pd.read_csv('C://Users/test/Documents/BackupPreviousWindows/UGroup/data_engineering_dataset.csv', sep=',', error_bad_lines=False)

train.head()

### Somehow there are 30 cloumns some unnamed columns popped into, hence this needs further cleaning

df22=train.iloc[:,0:17] 


### Test the Queries to make sure it is accurate before passing it into the Restful API
### Ability to fetch aggregate data on a dimension of your choice. Ideas are: number of vehicles purchased per store

q11= """SELECT count(VIN) as TotalPurchased, LocationNum FROM df1 group by LocationNum"""

print(ps.sqldf(q11, locals()))

### Filtering based on price and store location average purchase price per store, sorted list of most commonly sold makes/brands, etc.

q2 = """SELECT sum(PurchVal) as Total, LocationNum FROM df1 group by LocationNum order by Purchval desc"""

print(ps.sqldf(q2, locals()))

### Filtering based on price and store location average purchase price per store
q3 = """SELECT avg(PurchVal) as AvgPrice, LocationNum FROM df1 group by LocationNum order by LocationNum asc"""

print(ps.sqldf(q3, locals()))

### Filtering based on price and store location average purchase price per store orted list of most commonly sold brands
q4 = """SELECT avg(PurchVal) as AvgPrice, Brand, LocationNum FROM df1 group by LocationNum, Brand order by LocationNum asc"""

print(ps.sqldf(q4, locals()))


### Filtering based on price and store location average purchase price per store orted list of most commonly sold Make
q5 = """SELECT avg(PurchVal) as AvgPrice, Make, LocationNum FROM df1 group by LocationNum, Make order by LocationNum asc"""

print(ps.sqldf(q5, locals()))


#### Make sure to ensure this is cleaned

df22.columns = df22.columns.str.strip()

### Create a SQLite database for Car

con = sqlite3.connect("car3.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
df22.to_sql("car3", con, if_exists='replace')

### Let's create the cursor object
cur = con.cursor()
## Once we have a Cursor object, we can use it to execute a query against the database with the aptly named execute method. The below code will fetch the first 5 rows from the airlines table:

### Test to make sure that the data can be selected from the table
cur.execute("select * from car3 limit 5;")

results = cur.fetchall()
print(results)


### Now connect with the sqlite car database
db_connect = create_engine('sqlite:///car3.db')
app = Flask(__name__)
api = Api(app)


#### Ability to fetch aggregate data on a dimension of your choice. Ideas are: number of vehicles purchased per store
class PurchaseValuePerLoc(Resource):
    def get(self):
        conn = db_connect.connect() # connect to database
        query = conn.execute("SELECT count(VIN) as TotalPurchased, LocationNum FROM car3 group by LocationNum order by LocationNum Asc") # This line performs query and returns json result
        return {'PurchaseValuePerLoc': [i[0] for i in query.cursor.fetchall()], }
     
        

### Filtering based on price and store location average purchase price per store
class AvgPurch(Resource):
    def get(self):
        conn = db_connect.connect()
        query = conn.execute("SELECT avg(PurchVal) as AvgPrice, LocationNum FROM car3 group by LocationNum order by LocationNum dsc;")
        return {'AvgPurch': [i[0] for i in query.cursor.fetchall()], }


### Filtering based on price and store location average purchase price per store orted list of most commonly sold brands
class AvgPurch_Brand(Resource):
    def get(self):
        conn = db_connect.connect()
        query = conn.execute("SELECT avg(PurchVal) as AvgPrice, Brand, LocationNum FROM car3 group by LocationNum, Brand order by LocationNum asc ")
##      result = {'data': [dict(VIN(tuple (query.keys()) ,i)) for i in query.cursor]}
        return {'AvgPurch_Brand': [i[0] for i in query.cursor.fetchall()]}
    


### Filtering based on price and store location average purchase price per store orted list of most commonly sold Make
class AvgPurch_Make(Resource):
    def get(self):
        conn = db_connect.connect()
        query = conn.execute("SELECT avg(PurchVal) as AvgPrice, Make, LocationNum FROM car3 group by LocationNum, Make order by LocationNum asc ")
##      result = {'data': [dict(VIN(tuple (query.keys()) ,i)) for i in query.cursor]}
        return {'AvgPurch_Make': [i[0] for i in query.cursor.fetchall()]}
        

api.add_resource(PurchaseValuePerLoc,'/PurchaseValuePerLoc') # Route_1
api.add_resource(AvgPurch, '/AvgPurch') # Route_2
api.add_resource(AvgPurch_Brand, '/AvgPurch_Brand') # Route_3
api.add_resource(AvgPurch_Make, '/AvgPurch_Make') # Route_4

if __name__ == '__main__':
     app.run(port='5002')
    


con.close()









