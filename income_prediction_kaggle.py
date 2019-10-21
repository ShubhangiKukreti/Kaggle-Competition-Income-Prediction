#importing libraries
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt

def FeatureExtractionMethod(dataFrame,columnName):

    encoder = OneHotEncoder(sparse = False, handle_unknown = 'ignore')

    #reshaping the column
    temp = dataFrame[columnName]
    temp = np.array(temp).reshape(-1,1)

    #Extract and join the data frame
    dataFrame = dataFrame.join(pd.DataFrame(encoder.fit_transform(temp),columns=encoder.categories_,index=dataFrame.index))

    #Remove the Column
    dataFrame = dataFrame.drop([columnName], axis = 1)

    return dataFrame

def SpecialFeaturesMethod(uniques,dataFrame,columnName):
    
    encoder = OneHotEncoder(categories = [uniques],sparse = False, handle_unknown = 'ignore')

    #reshaping the column
    temp = dataFrame[columnName]
    temp = np.array(temp).reshape(-1,1)
   
    #Extract the column and join the data frame
    dataFrame = dataFrame.join(pd.DataFrame(encoder.fit_transform(temp),columns=encoder.categories_,index=dataFrame.index))

    #Remove the profession Column
    dataFrame = dataFrame.drop([columnName], axis = 1)

    return dataFrame


#loading data
data = pd.read_csv('training-data-with-labels.csv')
data_test = pd.read_csv('test-data-without-labels.csv')

# removing outliers from training data
data = data[(data['Income in EUR']>0) & (data['Income in EUR']<2600000)]

#filling missing numeric values for training data
data['Year of Record'].fillna(data['Year of Record'].mean(), inplace=True)   
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Gender'] = data['Gender'].fillna('unknown gender')
data['Profession'] = data['Profession'].fillna('Unknown Profession')
data['University Degree'] = data['University Degree'].fillna('No degree')
data['Hair Color'] = data['Hair Color'].fillna('Unknown hair')

#filling missing numeric values for test data
data_test['Year of Record'].fillna(data_test['Year of Record'].mean(), inplace=True)   
data_test['Age'].fillna(data_test['Age'].mean(), inplace=True)
data_test['Gender'] = data_test['Gender'].fillna('unknown gender')
data_test['Profession'] = data_test['Profession'].fillna('unknown Profession')
data_test['University Degree'] = data_test['University Degree'].fillna('No degree')
data_test['Hair Color'] = data_test['Hair Color'].fillna('Unknown hair')

#replacing unknowns with values for training data
data['Gender'] = data['Gender'].replace(['0','unknown' ], 'unknown gender')                   
data['University Degree'] = data['University Degree'].replace(['0','#N/A'], 'No degree')
data['Hair Color'] = data['Hair Color'].replace(['0','unknown'], 'Unknown Hair')

#replacing 0s with categorical values for test data
data_test['Gender'] = data_test['Gender'].replace(['0','unknown'], 'unknown')           
data_test['University Degree'] = data_test['University Degree'].replace(['0','#N/A'], 'No degree')
data_test['Hair Color'] = data_test['Hair Color'].replace(['0','unknown'], 'Unknown')

#one hot encoding
data_unique_prof = data['Profession'].unique()     #getting uniques for Profession
data_unique_country = data['Country'].unique()     #getting uniques for Country
data = FeatureExtractionMethod(data, 'Gender')
data = FeatureExtractionMethod(data, 'University Degree')
data = FeatureExtractionMethod(data, 'Hair Color')
data = SpecialFeaturesMethod(data_unique_prof, data, 'Profession')
data = SpecialFeaturesMethod(data_unique_country, data, 'Country')

data_test = FeatureExtractionMethod(data_test, 'Gender')
data_test = FeatureExtractionMethod(data_test, 'University Degree')
data_test = FeatureExtractionMethod(data_test, 'Hair Color')
data_test = SpecialFeaturesMethod(data_unique_prof, data_test, 'Profession')
data_test = SpecialFeaturesMethod(data_unique_country, data_test, 'Country')

#splitting data for input and output
X = data.drop(['Income in EUR'], axis = 1)
y = data['Income in EUR']

#splitting data for training and validation
xTrain, xValidate, yTrain, yValidate = train_test_split(X, y, test_size = 0.2, random_state = 0)

#extracting test data 
X_test =  data_test.drop("Income", axis=1)

#preforming Ridge Regression
regressor = BayesianRidge()
regressor.fit(xTrain, yTrain)
y_pred = regressor.predict(xValidate)

#Ridge Regressin on test data
result = regressor.predict(X_test)

#putting data in output file
res = pd.DataFrame(X_test['Instance'])
res['Income'] = result
res.index = X_test.index        # for comparison
res.to_csv("tcd ml 2019-20 income prediction submission file.csv")

# calculating the rmse
rms = np.sqrt(mean_squared_error(yValidate, y_pred))
print("rmes is" + str(rms))


