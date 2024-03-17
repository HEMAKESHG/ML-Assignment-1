# ASSIGNMENT - LINEAR REGRESSION:

## Name: Hemakesh G
## Register No.: 212223040064

## Q1. Create a scatter plot between cylinder vs Co2Emission (green color)

### Program:
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('FuelConsumption (1).csv')

plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green')
plt.xlabel('CYLINDERS')
plt.ylabel('CO2EMISSIONS')
plt.title('Cylinder vs CO2 Emission')
plt.show()
```
### Output:
![mlq1](https://github.com/HEMAKESHG/ML-Assignment-1/assets/144870552/d9061b82-9cee-4b86-b04a-ca88c72bc8bf)


## Q2. Using scatter plot compare data   cylinder vs Co2Emission and Enginesize Vs Co2Emission using different colors

### Program:
```
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green', label='Cylinder')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='red', label='Engine Size')
plt.xlabel('Feature')
plt.ylabel('CO2 Emission')
plt.title('Comparison of Cylinder and Engine Size vs CO2 Emission')
plt.legend()
plt.show()
```
### Output:
![mlq2](https://github.com/HEMAKESHG/ML-Assignment-1/assets/144870552/ae63bf6a-a492-4c7a-aa03-9237c1a043d5)


## Q3. Using scatter plot compare data   cylinder vs Co2Emission and Enginesize Vs Co2Emission and FuelConsumption_comb Co2Emission using different colors

### Program:
```
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green', label='Cylinder')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='red', label='Engine Size')
plt.scatter(df['FUELCONSUMPTION_COMB'], df['CO2EMISSIONS'], color='blue', label='Fuel Consumption')
plt.xlabel('Feature')
plt.ylabel('CO2 Emission')
plt.title('Comparison of Cylinder, Engine Size, and Fuel Consumption vs CO2 Emission')
plt.legend()
plt.show()
```
### Output:
![mlq3](https://github.com/HEMAKESHG/ML-Assignment-1/assets/144870552/f8e71d68-8e62-455b-9435-accb985698b2)


## Q4. Train your model with independent variable as cylinder and dependent variable as Co2Emission

### Program:
```
X = df[['CYLINDERS']]
y = df['CO2EMISSIONS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_fuel = LinearRegression()
model_fuel.fit(X_train, y_train)

plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, model_fuel.predict(X_test), color='red', label='Regression Line')
plt.xlabel('Cylinder')
plt.ylabel('CO2EMISSION')
plt.title('CYLINDER vs CO2 EMISSION')
plt.legend()
plt.show()
```
### Output:
![image](https://github.com/HEMAKESHG/ML-Assignment-1/assets/144870552/0029d816-5145-4b26-aa3c-a3e593f53dd1)



## Q5. Train another model with independent variable as FuelConsumption_comb and dependent variable as Co2Emission

### Program:
```
data = pd.read_csv("/content/FuelConsumption.csv")

X = data['FUELCONSUMPTION_COMB'].values.reshape(-1, 1)
y = data['CO2EMISSIONS'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_fuel = LinearRegression()
model_fuel.fit(X_train, y_train)

plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, model_fuel.predict(X_test), color='red', label='Regression Line')
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel('CO2EMISSIONS')
plt.title('Fuel Consumption vs Co2 Emission')
plt.legend()
plt.show()

```
### Output:
![image](https://github.com/HEMAKESHG/ML-Assignment-1/assets/144870552/454e18f1-d311-4e0b-9ae6-2e01113ef2c1)


## Q6. Train your model on different train test ratio and train the models and note down their accuracies

### Program:
```
X = df[['CYLINDERS']]
y = df['CO2EMISSIONS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_cylinder = LinearRegression()
model_cylinder.fit(X_train, y_train)
```
### Output:
![Screenshot 2024-03-08 162816](https://github.com/HEMAKESHG/ML-Assignment-1/assets/144870552/3a0eeebd-c12f-409a-b206-ca48af3c45ea)

