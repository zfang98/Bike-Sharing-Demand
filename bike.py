import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics,svm
import xgboost as xg
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from scipy.stats.stats import pearsonr


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
print (train.head(3))
print (train.info())


train.corr()
sns.boxplot(train['hour'],train['count'])
plt.show()
train=train[np.abs(train['count']-train['count'].mean())<=3*train['count'].std()]
sns.boxplot(train['count'])
plt.show()
# features workingday, weather, season, holiday are categorical features
print(train['workingday'].value_counts())
print(train['holiday'].value_counts())
print(train['weather'].value_counts())
print(train['season'].value_counts())
#transforming datetime from text formate to standard datetime formate
train['datetime']=train['datetime'].apply(pd.to_datetime)
train['month']=train['datetime'].apply(lambda x:x.month)
train['year']=train['datetime'].apply(lambda x:x.year)
train['hour']=train['datetime'].apply(lambda x:x.hour)

# sns.barplot(x='hour',y='count',data=train,estimator=sum)
# investigate correlation between weather features

weather_features=train[['weather','humidity','temp','atemp','windspeed']]
sns.heatmap(weather_features.corr())
# investigate correlation between all features
sns.heatmap(train.corr())
# hihgly correlated features: season/month, temp/atep, registered/count
#registered and causal are features that not exist in test data. should be removed when used for training models
# one the other hand, month and atep were chosen as training features.
# investigating the values in feature 'windspeed'
plt.hist(test['windspeed'],bins=30)
plt.show()
# use randomforst to fit in 0 wind speed

wind0=train[train['windspeed']==0]
windnot0=train[train['windspeed']!=0]

rg_wind=RandomForestRegressor()
windcolumns=['atemp','humidity','month','season']
rg_wind.fit(windnot0.loc[:,windcolumns],windnot0['windspeed'])
wind0values=rg_wind.predict(wind0.loc[:,windcolumns])
wind0['windspeed']=wind0values
train=windnot0.append(wind0)
train.reset_index(inplace=True)


# plt.hist(train['windspeed'],bins=30)
# plt.show()
new_train=train.drop(['casual','season','registered','temp','datetime'],axis=1)

# print (list(new_train))
#generate dummies for month, weather, hour, year,
year=pd.get_dummies(new_train['year'],prefix='y',drop_first=True)
month=pd.get_dummies(new_train['month'],prefix='m',drop_first=True)
hour=pd.get_dummies(new_train['hour'],prefix='h',drop_first=True)
weather=pd.get_dummies(new_train['weather'],prefix='w',drop_first=True)
new_train=new_train.join(year)
new_train=new_train.join(month)
new_train=new_train.join(hour)
new_train=new_train.join(weather)
#delete original features
del new_train['year']
del new_train['month']
del new_train['hour']
del new_train['weather']
# print (list(new_train))
y=new_train['count'].values
y=np.log1p(y)
final_train=new_train.drop(['count'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(final_train,y,test_size=0.20)

# use different algorithms to predict y: linear regression, polylinear regression, randomforest, xgboost,svm
#random forest
#when n_estimator=100, error=0.44
#when n_estimator=500, error=0.452
#when n_estimator=1000, score=0.9056
#---------
#RMSLE
# print (list(final_train))
#
def rmsle(truth, predict):
    log_diff=(truth-predict)**2
    error=np.sqrt(log_diff.mean())
    return error
rmse = make_scorer(rmsle, greater_is_better=False)
#-------------

rf=RandomForestRegressor(n_estimators=1000,random_state=0)
rf.fit(X_train,y_train)
print (rf.feature_importances_)
pred=rf.predict(X_test)
error=rmsle(y_test, pred)
print (error)
#---------------
# linear regression error=0.576
from sklearn .linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
error=rmsle(y_test, pred)
print (error)
# #-----------------------
# support vestor regression  error=0.997
clf=svm.SVR()
clf.fit(X_train,y_train)
pred=clf.predict(X_test)
error=rmsle(y_test, pred)
print (error)
#-------------
## xgboost when n_estimator=100, error=0.537, when n_estimator=500,error=0.533,
xgb_regre = xg.XGBRegressor(booster='gbtree',random_state=0)
sample_ratio=[0.8]
L1_regularization=[0.05,0.01]
estimator=[100]
max_depth=[4]
learning_rate=[0.15]
features_ratio=[0.6,0.5]

XGB_grid = {'n_estimator':estimator,
           'learning_rate':learning_rate,
           'subsample':sample_ratio,
           'colsample_bytree':features_ratio,
           'max_depth':max_depth,
           'reg_alpha':L1_regularization,
           }

XGB_random = RandomizedSearchCV(estimator = xgb_regre, param_distributions = XGB_grid,
                                n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1,scoring=rmse)

XGB_random.fit(X_train,y_train)
xgb_predict=XGB_random.predict(X_test)
r_value,p_value = pearsonr(y_test,xgb_predict)
error=rmsle(y_test,xgb_predict)
print ('Pearson correational coefficiant is '+str(round(r_value,2)))
print ('Root mean squared error is '+str(round(error,2)))

plt.scatter(y_test,xgb_predict)
plt.show()

y_pred=np.exp(xgb_predict)-1
print (y_pred)
