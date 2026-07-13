# Homework-
# Polynomial Regression -> Data from Kaggle -> Use Seabon -> Dump Data Pickel

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Data-
url = "https://raw.githubusercontent.com/hxrsh001/student-dataset/main/student_dataset_10000_rows.csv"
df = pd.read_csv(url)

print("Original Dataset: ")
print(df, "\n")

x = df[["study_hours"]]
y = df["exam_score"]

# Poly 
poly = PolynomialFeatures(degree=5)
x_poly = poly.fit_transform(x)

# Train Tst Split-
x_train, x_test, y_train, y_test = train_test_split(x_poly,y, test_size=0.2, random_state=42)

# Model Train-
model = LinearRegression()
model.fit(x_train, y_train)

# Get New Data-
new_data = int(input("Enter Hours Studied: "))
new_data = pd.DataFrame({
    "study_hours": [new_data]
})

new_data_poly = poly.transform(new_data)

# Prediction-
prediction = model.predict(new_data_poly)
print(f"Exam Score: {prediction[0]}")


# Visuals Using Seaborn- 
sns.scatterplot(
    data=df,
    x="study_hours",
    y="exam_score"
)
plt.title("Study Hours vs Exam Score")
plt.show()

# Dump Data Using Pickle-
with open("PolynomialRegression.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model Saved Succesfully!")