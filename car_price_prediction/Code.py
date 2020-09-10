#miniproject Regression [car price prediction]
import numpy as np
import pandas as pd

#import data set
dataset = pd.read_csv("car_data.csv")
#explore and understand the data 
dataset.info()
#explore the categorical some columns 
dataset['transmission'].value_counts()
#prepare wanted independent variables
x= dataset.iloc[:,[1,3,4,6]].values
#prepare dependent variable
y=dataset.iloc[:,[2]].values

#preprocessing phase 
#encode categorical data using LabelEncoder (this can be changed to one hot encoder or others )

from sklearn.preprocessing import LabelEncoder
Label1=LabelEncoder()
Label2=LabelEncoder()
x[:,2]=Label1.fit_transform(x[:,2])
x[:,3]=Label2.fit_transform(x[:,3])

print(x.shape)

#split data set into training and testing set 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2 ,random_state=0)

# creating and fitting the the training set (here you can choose your model e.g. multiple Linear Regrssion, support vector regression..etc)
# instantiating a random forest regression model
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0) #n_estimator is one of the hyperparameters that needs to be tuned. you can tune it experimentally  
regressor.fit(x_train,y_train)

#calculate accurecy of your model
accuracy=regressor.score(x_test,y_test)
print(f'accuracy ={accuracy*100}')

#predicting new data
newdata= ['2017','7000','Petrol','Manual']
#encode new data
newdata[2]=Label1.transform([newdata[2]])[0]
newdata[3]=Label2.transform([newdata[3]])[0]
regressor.predict([newdata])
