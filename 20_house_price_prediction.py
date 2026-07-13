# Project using MySql, Joblib and SLR | MultiLinearRegression-

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

'''data = {
    "name": ["Harsh", "Rajendra", "Dheeraj", "Yash"],
    "avg_income": [1000, 200, 300, 400],
    "house_age": [10.2, 2.5, 3.0, 4.0],
    "num_rooms": [3, 4, 5, 1],
    "price": [10000, 7000, 15000, 11000]
}'''

# MySQL Connector-
import pymysql

conn = pymysql.connect(host="13.220.156.88",user="mluser",database="ml_db",password="mlpass123")

query = "select avg_income,house_age,num_rooms,price from houses"

df = pd.read_sql(query, conn)
print(df)

#Data Frame-
#df = pd.DataFrame(data)

# Handle Missing Values-
#df.drop("name", axis=1, inplace=True)

# Feature Scaling-
scaler = StandardScaler()
df[['avg_income', 'house_age', 'num_rooms']] = scaler.fit_transform(df[['avg_income', 'house_age', 'num_rooms']])
print(df, "\n")

# Split Data-
x = df[['avg_income', 'house_age', 'num_rooms']]
y = df[['price']]

# Train Test Split-
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=2)

# Model Train-
model = LinearRegression()
model.fit(x_train, y_train)

# Prediction Data-
new_data = pd.DataFrame({
    "avg_income": [900],
    "house_age": [8],
    "num_rooms": [5]
})

# Scale new data

new_data = pd.DataFrame({
    "avg_income": [900],
    "house_age": [8],
    "num_rooms": [5]
})

new_data_scaled = scaler.transform(new_data)

new_data_scaled = pd.DataFrame(
    new_data_scaled,
    columns=["avg_income", "house_age", "num_rooms"]
)

prediction = model.predict(new_data_scaled)

print("Predicted House Price:", prediction[0])

# Model Dump Using Joblib-
import joblib
#joblib.dump(model, "house_model.joblib")