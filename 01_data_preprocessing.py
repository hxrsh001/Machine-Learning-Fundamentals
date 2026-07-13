# Data Preprocessing-

import pandas as pd

data = {
    'Name' : ["Ram", "Kamal", "Ajay"],
    'Age' : [25, None, 32],
    'Salary' : [5000, 6000, None]
}

df = pd.DataFrame(data)
print("Original DataFrame: ")
print(df)

# isnull()-
print("\n", df.isnull())
print("\n", df.isnull().sum())

# dropna()-
new_df = df.dropna()
print(new_df, "\n")

# dropna() when specific data is missing-
#new_df = df.dropna(subset=['Age'])
#print(new_df, "\n")

new_df = df.dropna(subset=['Salary'])
print(new_df, "\n")

# fillna()-
avg_age = df['Age'].median()
df["Age"] = df["Age"].fillna(avg_age)
#df["Age"].fillna(df["Age"].mean(), inplace=True)
print(df, "\n")

df["Salary"] = df["Salary"].fillna(10000)
print(df)