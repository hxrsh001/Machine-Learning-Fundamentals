import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Data-
data = ({
    "study_hour" : [8, 5, 2, 9, 10, 12, 3, 4, 1],
    "marks": [85, 63, 22, 93, 95, 99, 33, 37, 7],
    "result": [0, 0, 1, 0, 0, 0, 0, 1, 1]
})

df = pd.DataFrame(data)

x = df[['study_hour', 'marks']]
y = df['result']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Prediction: ", y_pred)
print("Accuracy: ", accuracy_score(y_test, y_pred))

plt.figure(figsize=(8,5))

plt.scatter(
    df["study_hour"],
    df["marks"],
    c=df["result"],      # color according to class
    cmap="bwr",          # Blue = 0, Red = 1
    s=100
)

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.colorbar(label="Result")
plt.grid(True)

plt.show()
