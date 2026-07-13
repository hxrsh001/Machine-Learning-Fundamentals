# Ridge Regression-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = {
    "size": [1000,1200,1500,1800,2000,2200,2500,2700,3000,3200],
    "bedrooms": [2,2,3,3,3,4,4,4,5,5],
    "age": [20,18,15,12,10,8,6,5,4,2],
    "price": [30,35,45,50,55,60,68,72,80,85]
}
df = pd.DataFrame(data)
print(df, "\n")

# Split Data in x and y-
x = df[['size', 'bedrooms', 'age']]
y = df['price']

# Training & Testing-
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# Ridge Resgression-
ridge = Ridge(alpha=1.0)
ridge.fit(x_train,y_train)

# Prediction-
y_pred = ridge.predict(x_test)
print(f"X Test: \n {x_test}")
print(f"Predicted data: {y_pred}\n")

# MAE (Mean Absolute Error)-
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# MSE-
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}/n")