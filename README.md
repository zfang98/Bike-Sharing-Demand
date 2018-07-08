# Bike-Sharing-Demand
###
#### Description: biking is good for the environment and people's health. Study has shown that, because there are more people cycling in the city of Vancouver, the expenses on health care had been distinctively decreaed. I think cycling should be encouraged when both the distance and time are allowed for the purpose of exercising and transportation. 
###
####  This project use history data to predict how many bikes will be in demand in every hour of a day.
###### Documents: Train.csv: Data used for training models
######            Test.csv: Data used for testing models.
###### Features included in data:
###### Weather: _categorical-
###### datetime: _numeric_  
###### season: _categorical_
###### holiday: _categorical_
###### working day: _categorical_
###### temp: _numeric_
###### atemp: _numeric_
###### humidity: _numeric_
###### windspeed: _numeric_
###### casual: _numeric_
###### registered: _numeric_
###### count: _numeric_

#### Metric used to evaluate the models: Root Mean Squared Logarithmic Error (RMSLE)



###### Softwares and operation system: Anaconda3, Python 3.7, OSX 10.10.5
##### Contribution: investigted distribution of numerical features, filled missing data, checked outliers, and deleted unuseful features. Applied multiple algorithms from Scikit learn: Regression, Support Vector Machine(SVM), Random Forest, K-Nearest Neighbors (KNN), XGBoost to predict bike sharing demand.



