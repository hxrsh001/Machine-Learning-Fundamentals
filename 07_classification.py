# Classification-

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

df = pd.DataFrame({
    "Weight": [120,140,150,170],
    "Size": [6,7,7,8]
})
x = df
y = ["Apple", "Apple", "Orange", "Orange"]

# Model-
model = DecisionTreeClassifier()
model.fit(x,y)

# Prediction Data-
input_weight = int(input("Enter the Weight: "))
input_size = int(input("Enter the Size: "))
new_data = pd.DataFrame({
    "Weight": [input_weight],
    "Size": [input_size]
})

# Prediction:
prediction = model.predict(new_data)
print(prediction)