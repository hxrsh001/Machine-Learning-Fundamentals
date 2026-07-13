import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

data = {
    "age": [20, 18, np.nan, 26, 30],
    "gender": ["male", "female", "others", "male", None],
    "name": ["dheeraj", "kavi", "sapu", "yash", "hello"]
}

df = pd.DataFrame(data)
print("Original Value: ")
print(df, "\n")

# Handle Missing Values-
df["age"] = df["age"].fillna(df["age"].mean())
df.dropna(subset=["gender"], inplace=True)
print(df, "\n")

# Label Encoding-
le = LabelEncoder()
df["gender"] = le.fit_transform(df["gender"])
print(df, "\n")

# One Hot Encoding-
oe = OneHotEncoder()
oe_e = oe.fit_transform(df[["gender"]]).toarray()
print(oe_e)