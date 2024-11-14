# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle


# Get data from csv file

data = pd.read_csv('/Users/thanhduong/Workspace/Predict Student Score/Student_Performance.csv')


# Data cleaning

x = data.iloc[:, :-1]               # Get the input data

y = data.iloc[:,[-1]]               # Get the label data

# Because in Extracurricular Activities column, we have "Yes" and "No" (string data format) which is not used to train model
# So, I have to convert it to 0 and 1 meaning that 0 is false and 1 is true 
le = LabelEncoder()
x['Extracurricular Activities'] = le.fit_transform(x['Extracurricular Activities']).astype(int)

# get the train and test data for x and y
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 10)


# Train model
reg = LinearRegression()
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)
result = np.concatenate((y_test[0:5],y_pred[0:5]), axis=1).round()

# Create a model 
pickle.dump(reg, open('model.pkl', 'wb'))
