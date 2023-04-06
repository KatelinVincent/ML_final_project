import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

"""
FIRST: 
Linear Regression
Multiple Linear Regression
KNN Regression
Decision Tree
SVM
Gradient Boosting
Random Forest

HYPERPARAMETER TUNING : Bayesian Optimization

THEN:
Random Forest WITH Permutation Importance
Random Forest WITH Feature Importance
Decision Tree WITH Variance Threshold
Gradient Boosting WITH Permutation feature importance
Gradient Boosting WITH Recursive Feature Elimination
Linear Regression WITH LASSO

PERFORMANCE METRICS:
Mean Absolute Error
R-Squared
Mean Square Error

"""

train_data_path = './Data/train.csv'
test_data_path = './Data/test.csv'

#load data in
df_train = pd.read_csv(train_data_path, sep=",")
df_test = pd.read_csv(test_data_path, sep=",")

# training target variable array
y = np.array(df_train['SalePrice'].values)

# training feature variable matrix
X = np.matrix(df_train.iloc[:, :80].values)

#split training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)

# final test
X_final_test = np.matrix(df_test.iloc[:, :].values)


#create steps for pipeline - hyper parameter tuning

#create pipeline

#fit each model

#predict with each model

#print Model + performance metrics for each model

#use most efficient model to run the final test

#print final predictions for train.csv
