# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:27:49 2022

@author: harini
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:05:40 2022

@author: harini
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as pt 
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as sm
ds=pd.read_csv(r"C:\Users\harini\OneDrive\Documents\city_day.csv")
print(ds.head(5))
print(ds.info())
print(ds.isna())

ds['Date']=pd.to_datetime(ds['Date'])      
print(ds.info())
print(ds.isna().sum())
print(ds.describe())
df=ds.copy()

pt.figure(figsize=(30,10))
msno.bar(ds)
pt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True)
pt.show()
msno.heatmap(ds)
pt.show()


"""remove xylene attriibute as there are many null values"""
ds=df.dropna(axis=0,thresh=ds.shape[1]-13)
ds=ds.drop(labels="Xylene",axis=1)
print(ds.head())
print(ds.info())

d=pd.DataFrame()
sns.boxplot(data=ds,orient=(90))
pt.figure(figsize=(100,80))
x=pd.DatetimeIndex(ds['Date']).year
def find_outliers_IQR(ds):
   q1=ds.quantile(0.25)
   q3=ds.quantile(0.75)
   IQR=q3-q1
   outliers = ds[((ds<(q1-1.5*IQR)) | (ds>(q3+1.5*IQR)))]
   return outliers
for i in range(2,len(ds.columns)-1):
    outliers = find_outliers_IQR(ds[ds.columns[i]])
    print("number of outliers: "+ str(len(outliers)))
    print("maximum outlier value: "+ str(outliers.max()))
    print("minimum outlier value: "+ str(outliers.min()))
    print("average:",outliers.mean())
   
drop_outlier = ds[(ds['AQI']>600) | (ds['PM2.5']>200) | (ds['NO']>70) |(ds['NH3']>50)
                  | (ds['NO2']>90) | (ds['NOx']>100) | 
                  (ds['PM10']>400) |(ds['CO']>30)|(ds['SO2']>60) | (ds['O3']>125) | (ds['Benzene']>30) 
                  | (ds["Toluene"]>50)].index
ds=ds.drop(drop_outlier)
print(ds.describe())
city_uni=(ds.iloc[:,0]).unique()
dff=df.copy()
dff.info()

for j in range(2,14):
    b=pd.DataFrame();
    for i in range(len(city_uni)):
        a=(((ds.loc[ds["City"]==city_uni[i]]).iloc[:,j]))

        if(np.isnan(a.median()) or a.median()==0):
            a=a.fillna((ds.iloc[:,j].median()))
        else:    
            a=a.fillna(a.mean())
        b=pd.concat([b,a],axis=0)
    ds[ds.columns[j]]=b;    
ds.info()
ds.isna().sum()


lrd=(ds[ds["AQI_Bucket"].isna()])
lrd1=ds[ds["AQI_Bucket"].notnull()]
x_train=lrd1.iloc[:,2:14]
y_train=lrd1.iloc[:,14]
x_test=lrd.iloc[:,2:14]
y_test=lrd.iloc[:,14]
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_test1=pd.DataFrame(lr.predict(x_test))
y_test1.index=y_test.index

aqi=pd.concat([y_test1,y_train],axis=0)
aqi.sort_index()
ds["AQI_Bucket"]=aqi
aqi_uni=(ds.iloc[:,14]).unique()
for i in aqi_uni:
    print(i,len(ds.loc[ds["AQI_Bucket"]==i]))
    

scale= StandardScaler()
x = ds.iloc[:,2:13]
nscale=x
y = ds.AQI
scaled_data = scale.fit_transform(x) 
ds_f = pd.DataFrame(scaled_data)
ds.iloc[:,2:13]=scaled_data
a={"Severe":0,"Very Poor":1,"Poor":2,"Moderate":3,"Satisfactory":4,"Good":5}
ds["AQI_Bucket"]=ds["AQI_Bucket"].map(a)
print(ds.head())
print(ds.info())
df=ds.copy()

from sklearn.model_selection import train_test_split
x=ds.iloc[:,2:13]
y=ds.iloc[:,13]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) 

def calc_metr(y_test,pred):
    
   mae =round(sm.mean_absolute_error(y_test, pred), 3) 
   mse=round(sm.mean_squared_error(y_test, pred), 3) 
   rmse=round(np.sqrt(mse),3)
   r2= round(sm.r2_score(y_test, pred), 3)
   return([mae,mse,rmse,r2]) 

final_metric=[]
from sklearn import linear_model
model=linear_model.LinearRegression()
model.fit(x_train,y_train)
pred=model.predict(x_test)
lr_m=calc_metr(y_test,pred)
print(lr_m)
final_metric.append(lr_m)

from sklearn.tree import DecisionTreeRegressor 
regressor = DecisionTreeRegressor(random_state = 0) 
regressor.fit(x_train, y_train)
pred2=model.predict(x_test)
lr_m=calc_metr(y_test,pred2)
print(lr_m)
final_metric.append(lr_m)

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=5,random_state=0)
regressor.fit(x_train,y_train)
pred=regressor.predict(x_test)
lr_m=calc_metr(y_test,pred)
print(lr_m)
final_metric.append(lr_m)


from sklearn.linear_model import Ridge
rr = Ridge(alpha=0.01)
rr.fit(x_train, y_train) 
pred= rr.predict(x_test)
lr_m=calc_metr(y_test,pred)
print(lr_m)
final_metric.append(lr_m)

from sklearn.linear_model import Lasso
model_lasso = Lasso(alpha=0.01)
model_lasso.fit(x_train, y_train) 
pred= model_lasso.predict(x_test)
lr_m=calc_metr(y_test,pred)
print(model_lasso.coef_)
print(lr_m)
final_metric.append(lr_m)


from sklearn.linear_model import ElasticNet
model_enet = ElasticNet(alpha = 0.01)
model_enet.fit(x_train, y_train) 
pred= model_enet.predict(x_test)
lr_m=calc_metr(y_test,pred)
print(lr_m)
final_metric.append(lr_m)


print("------------------------------------------------------------------------")
print("MODEL                  :  MAE        MSE       RMSE    R-Squared(R2) ")
print("------------------------------------------------------------------------")
print("Linear Regression      :",final_metric[0][0],"   ",final_metric[0][1],"  ",final_metric[0][2],"  ",final_metric[0][3])
print("DecisionTree Regressor :",final_metric[1][0],"   ",final_metric[1][1],"  ",final_metric[1][2],"  ",final_metric[1][3])
print("RandomForest Regressor :",final_metric[2][0],"  ",final_metric[2][1],"  ",final_metric[2][2],"   ",final_metric[2][3])
print("Ridge Regression       :",final_metric[3][0],"   ",final_metric[3][1],"  ",final_metric[3][2],"  ",final_metric[3][3])
print("Lasso Regression       :",final_metric[4][0],"   ",final_metric[4][1],"  ",final_metric[4][2],"  ",final_metric[4][3])
print("ElasticNet  Regression :",final_metric[5][0]," ",final_metric[5][1],"  ",final_metric[5][2],"  ",final_metric[5][3])

predd=pd.DataFrame(pred2)
scaled_data = scale.fit_transform(predd) 
pred_f = pd.DataFrame(scaled_data)
pred_f.index=x_test.index
a=pd.concat([x_test,pred_f],axis=1)
a=a.rename({0 : 'AQI'}, axis=1)
pr=lr.predict(a)
print("predicted AQI:  " )
print(pred2)
print("Predicted AQI_Bucket")
print(pr)

