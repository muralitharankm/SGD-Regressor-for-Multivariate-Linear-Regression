# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, define features and targets
2.Split data into train and test sets, then standardize the features.
3.Train the multi-output regression model using SGDRegressor.
4.Predict and evaluate the model using Mean Squared Error, display results.
## Program:
```py
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: muralitharan k m 
RegisterNumber: 212223040121 
*/
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

dataset = fetch_california_housing()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['HousingPrice'] = dataset.target  
df['NumOccupants'] = df['AveOccup'] * 1.5  

print(df.head())

X = df.drop(columns=['HousingPrice', 'NumOccupants'])  
y = df[['HousingPrice', 'NumOccupants']]  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_regressor = MultiOutputRegressor(sgd_regressor)
multi_output_regressor.fit(X_train_scaled, y_train)


y_pred = multi_output_regressor.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print(f'Mean Squared Error for Housing Price: {mse[0]:.2f}')
print(f'Mean Squared Error for Number of Occupants: {mse[1]:.2f}')

predictions_df = pd.DataFrame({
    'ActualPrice': y_test['HousingPrice'].values,
    'PredictedPrice': y_pred[:, 0],
    'ActualOccupants': y_test['NumOccupants'].values,
    'PredictedOccupants': y_pred[:, 1]
})
print(predictions_df.head())
```

## Output:
![Screenshot 2024-09-25 085420](https://github.com/user-attachments/assets/3cf1f5d6-3fb0-45fb-8719-24728b2fc0d3)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
