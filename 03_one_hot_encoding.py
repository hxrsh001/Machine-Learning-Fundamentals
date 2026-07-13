# One Hot Encoding-

import pandas as pd

data = {
    'colors':['red','Blue','green','red','blue']
}
 
df = pd.DataFrame(data)
print("original data")
print(df)

# To make dataset lowercase-
df["colors"] = df['colors'].str.lower()
 
# one hot encoding
encoded_df = pd.get_dummies(
    df,
    columns=['colors'],
    dtype=int
)
 
print("after one hot encoding")
print(encoded_df)