# KNN (K-Nearest Neighbors)-

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data = {
    "weight": [100,110,120,170,180,190],
    "fruits": [0,0,0,1,1,1] # 0-apple, 1-orange
}

df = pd.DataFrame(data)
print(df, "\n")

x = df[["weight"]]
y = df["fruits"]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x,y)

new_data = pd.DataFrame({
    "weight": [1600]
})

pred_d = model.predict(new_data)
if pred_d == 0:
    print("Apple")
else:
    print("Orange")

# Classifications