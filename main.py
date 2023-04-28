import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier


train_data_path = './Data/train.csv'
test_data_path = './Data/test.csv'

"""
Regression algorithms: 
Linear Regression
Multiple Linear Regression
KNN Regression
Decision Tree
Random Forest
SVM
Gradient Boosting

Model evaluation metrics:
Root Mean Absolute Error
Adjusted R-Squared
Mean Square Error

Feature Selection Algorithms:
LASSO with Linear Regression - need standard scalar
RFE or SFS
Variance Threshold : Decision Tree, Linear Regression

Hyperparameter Tuning: Bayesian Optimization
"""

#load data in
df_train = pd.read_csv(train_data_path, sep=",")
df_test = pd.read_csv(test_data_path, sep=",")

# Preprocess categorical features
categorical_features = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 
                        'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
                        'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',
                         'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
                          'MiscFeature', 'SaleType', 'SaleCondition']

numeric_features = [col for col in df_train.columns if col not in categorical_features + ['SalePrice']]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Split into features and targets

y = df_train['SalePrice']
X = df_train.drop(columns=['SalePrice'])


# Preprocess the features
X = preprocessor.fit_transform(X)

# Get the column names for the one-hot encoded features
onehot_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(input_features=categorical_features)

# Combine the numeric and one-hot encoded feature names
all_feature_names = numeric_features + list(onehot_columns)
print(all_feature_names)
# Convert the transformed matrix X back into a DataFrame with the feature names. Pandas provides functions for explorarory data analysis.
X = pd.DataFrame(X, columns=all_feature_names)

print(X.head())

"""
# training target variable array
y = np.array(df_train['SalePrice'].values)

# training feature variable matrix
X = np.matrix(df_train.iloc[:, :80].values)

#split training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)

#create steps for pipeline - hyper parameter tuning

#create pipeline

#fit each model

#predict with each model

#print Model + performance metrics for each model

#use most efficient model to run the final test

#print final predictions for train.csv
"""