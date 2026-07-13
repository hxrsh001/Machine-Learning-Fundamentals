# Clustering-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    "age": [22,25,24,45,52,48,20,47],
    "income": [22000,25000,24000,45000,50000,48000,20000,47000]
}

df = pd.DataFrame(data)
print(df, "\n")

# Input | Feature
x = df[['age', 'income']]

# Model
model = KMeans(n_clusters=2, random_state=42)
model.fit(x)

# Clusters-
# clusters = model.labels_
# print(clusters, "\n")


df['clusters'] = model.labels_
print(df)

plt.scatter(df['age'], df['income'], c=df['clusters'])
plt.xlabel("AGE")
plt.ylabel("INCOME")
plt.title("Group The Data")
plt.grid()
plt.show()


