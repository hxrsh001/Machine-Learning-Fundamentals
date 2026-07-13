# Linear Regression -> UnderFitting-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.DataFrame({
    "level": [1,2,3,4,5]
})
x = df
y = [10,20,50,90,150]

# Model-
model = LinearRegression()
model.fit(x,y)

# Prediction data-
user_input = int(input("Enter the level: "))
new_data = pd.DataFrame({
    "level": [user_input]
})

prediction = model.predict(new_data)
print(f"Predicted Data: {prediction[0]}")

# Visual Data- 
plt.plot(x,y)
plt.title("Under Fitting")
plt.show()


print("\n")



# Polynomial Regression -> OverFitting-
z = pd.DataFrame({
    "label": [1,2,3,4,5]
})

# Poly Convert-
poly = PolynomialFeatures(degree=3)
z_poly = poly.fit_transform(z)

# Model-
model = LinearRegression()
model.fit(z_poly, y)

# Prediction Data-
input_data = int(input("Enter the label: "))
new1_data = pd.DataFrame({
    "level": [input_data]
})

# Input Data to Poly-
u_new1_data = poly.fit_transform(new1_data)

# Prediction-
predicted_data = model.predict(u_new1_data)
print(predicted_data[0])

# Visual Data- 
plt.plot(z_poly, y)
plt.title("Over Fitting")
plt.show()