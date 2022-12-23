# Importing the dependencies
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib as plt

# Importing the Dataset

heart_data=pd.read_csv('heart_disease_data.csv')
# Displaying the Dataset Information 
# print(heart_data.info())

# Displaying the Statistical Information
# print(heart_data.describe())

# Checking for Target Values
# print(heart_data['target'].value_counts())

# Initialising the data variables
# X -> input parameters
# Y -> output parameters
X=heart_data.drop('target',axis=1)
Y=heart_data['target']
# print(X)
# print(Y)

# Splitting the Data for Training and Testing
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y, random_state=4 )

# Using the Logistic Regression Model
model=LogisticRegression()
model.fit(X_train,Y_train)

#Predicting on Training Datasets
X_train_prediction=model.predict(X_train)
train_accuracy=accuracy_score(X_train_prediction,Y_train)
# print(train_accuracy)

#Predicting on Testing Datasets
X_test_prediction=model.predict(X_test)
test_accuracy=accuracy_score(X_test_prediction,Y_test)
# print(test_accuracy)

# Input from the User
age=int(input("Enter your Age "))
sex=int(input("Enter Gender 1=male,0=female "))
chest_pain_type=int(input("0=Heart Attack, 1=Angina, 2=Pericarditis, 3=Myocartidis "))
rest_bp=int(input("Enter resting blood pressure "))
serum_cholesterol=int(input("Enter Serum Cholesterol "))
fbs=int(input("Enter Fasting Blood Sugar >120mg/dl "))
restecg=int(input("Enter electrocardiographic results (0,1,2) "))
thalach=int(input("Enter Maximum Heart Beat "))
exang=int(input("Exercise Reduced Angina? 1=Yes, 0=No "))
oldpeak=float(input("Enter ST depression induced by exercise relative to rest "))
slope=int(input("Enter the slope of the peak exercise ST segment "))
ca=int(input("Enter number of major vessels (0-3) colored by flourosopy "))
thal=int(input("Enter Thal 1 = normal; 2 = fixed defect; 3 = reversable defect "))
inp=(age,sex,chest_pain_type,rest_bp,serum_cholesterol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)

# Creating a Prediction System
input_data=np.asarray(inp)
redefined_input=input_data.reshape(1,-1)
prediction=model.predict(redefined_input)
if prediction[0]==0:
     print("\nNo Heart Disease")
else: 
     print("\nHeart Disease")