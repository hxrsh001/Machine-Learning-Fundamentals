# Polynomial Regression-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Data-
df = pd.DataFrame({
    "Experience": [1,2,3,4,5]
})

x = df
y = [10,20,50,90,150]

# Poly Features-
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
print(x_poly)

# Model-
model = LinearRegression()
model.fit(x_poly, y)

# Get New Data-
input_data = int(input("Enter the Experience: "))
new_data = pd.DataFrame({
    "Experience": [input_data]
})

# Convert new data to Poly-
u_new_data = poly.fit_transform(new_data)

# Prediction-
prediction = model.predict(u_new_data)
print(prediction[0])


# Visuals-
plt.plot(x_poly, y)
plt.show()
