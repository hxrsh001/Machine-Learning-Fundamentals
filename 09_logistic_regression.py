# Logistic Regression- no algo only classification

from sklearn.linear_model import LogisticRegression
import pandas as pd
 
X = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6]
})
 
result = [0, 0, 0, 1, 1, 1]
 
model = LogisticRegression()
 
model.fit(X, result)

# New Data-
new_data = pd.DataFrame({
    "Hours":[5]
})
 
print(model.predict(new_data))