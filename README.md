# Developed by : Rama E.K. Lekshmi
# Register Number : 212222240082
# Date: 

# Ex.No: 07                                       AUTO REGRESSIVE MODEL

### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
```
Load the dataset
```py
data = pd.read_csv('AirPassengers.csv')
```
Convert 'Month' to datetime format and set it as the index
```py
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)
```
Check for stationarity using the Augmented Dickey-Fuller (ADF) test on '#Passengers'
```py
result = adfuller(data['#Passengers']) 
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```
Split data into training and testing sets (80% training, 20% testing)
```py
train_data = data.iloc[:int(0.8 * len(data))]
test_data = data.iloc[int(0.8 * len(data)):]
```

Define the lag order for the AutoRegressive model (adjust lag based on ACF/PACF plots)
```py
lag_order = 13
model = AutoReg(train_data['#Passengers'], lags=lag_order)
model_fit = model.fit()
```
Plot Autocorrelation Function (ACF) for '#Passengers'
```py
plt.figure(figsize=(10, 6))
plot_acf(data['#Passengers'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF) - #Passengers')
plt.show()
```
Plot Partial Autocorrelation Function (PACF) for '#Passengers'
```py
plt.figure(figsize=(10, 6))
plot_pacf(data['#Passengers'], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) - #Passengers')
plt.show()
```
Make predictions on the test set
```py
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
```
Calculate Mean Squared Error (MSE) for the test set predictions
```py
mse = mean_squared_error(test_data['#Passengers'], predictions)
print('Mean Squared Error (MSE):', mse)
```
Plot Test Data vs Predictions for '#Passengers'
```py
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data['#Passengers'], label='Test Data - #Passengers', color='blue', linewidth=2)
plt.plot(test_data.index, predictions, label='Predictions - #Passengers', color='orange', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('#Passengers')
plt.title('AR Model Predictions vs Test Data (#Passengers)')
plt.legend()
plt.grid(True)
plt.show()

```

### OUTPUT:

GIVEN DATA

![image](https://github.com/user-attachments/assets/a45571e6-b256-4b19-bba6-c01d604eed58)

PACF - ACF

![Untitled](https://github.com/user-attachments/assets/8e7e2365-5318-4984-9fbf-144014df8de1)

PREDICTION

![Untitled-1](https://github.com/user-attachments/assets/f128aef2-9127-4416-b84f-71a62e8bcb7f)

FINIAL PREDICTION

![image](https://github.com/user-attachments/assets/2e45b1d6-6b13-4a4c-8b0b-1a603d5fe95d)


### RESULT:
Thus we have successfully implemented the auto regression function using python.
