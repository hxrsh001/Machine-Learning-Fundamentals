'''# handle missing values
import pandas as pd
import numpy as np

data = {
    "colors":['red','green','blue','orange','green','blue',np.nan]
}
df = pd.DataFrame(data)
# print(df)
 
# handle null values
df.dropna(inplace=True)
print(df)
 
# label encoding
from sklearn.preprocessing import LabelEncoder
 
# method 1
le = LabelEncoder()
df['colors_encoder1'] = le.fit_transform(df['colors'])
 
# method 2
df['colors_encoder2'] = LabelEncoder().fit_transform(df['colors'])
# print(df)
 
# method 3
import sklearn
df['colors_encoder3'] = sklearn.preprocessing.LabelEncoder().fit_transform(df['colors'])
print(df)
 
# one hot encoding
from sklearn.preprocessing import OneHotEncoder
print(df)
 
# drop cols of label encoding
if 'colors_encoder' in df.keys():
  df.drop(df[['colors_encoder','colors_encoder1','colors_encoder2','colors_encoder3']],axis=1,inplace=True)
 
print(df, "\n")
 
# one hot encoding
encoder = OneHotEncoder()
encoded = encoder.fit_transform(df['colors'])
print(encoded.toarray())

# one hot encoding (by the help of pandas)
df_encoder = pd.get_dummies(df,columns=['colors'])
print(df_encoder)'''

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