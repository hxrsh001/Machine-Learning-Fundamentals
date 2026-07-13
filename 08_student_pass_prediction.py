# Decision Tree-

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

df = pd.DataFrame({
    "Marks": [20,25,30,35,40,50,60]
})
x = df
y = ["Fail", "Fail", "Fail", "Fail", "Pass", "Pass", "Pass"]

# Model-
model = DecisionTreeClassifier()
model.fit(x,y)

# Prediction Data-
input_marks = int(input("Enter the Marks: "))
new_data = pd.DataFrame({
    "Marks": [input_marks]
})

# Prediction:
prediction = model.predict(new_data)
print(prediction)