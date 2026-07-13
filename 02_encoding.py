# Encoding-

from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = {
    'Name' : ["Ram", "Kamal", "Ajay", "Neet-u", "Kavi", "Div"],
    'Age' : [25, None, 32, 28, 21, 22],
    'Salary' : [5000, 6000, None, 7000, 8000, 9000],
    'Gender' : ["Male", "Male", "Male", None, "Female", "Female"]
}

df = pd.DataFrame(data)
print("Original DataFrame: ")
print(df, "\n")

# Label Encoding-

le = LabelEncoder()
df_label = df.copy()
df_label["Gender"] = df["Gender"].fillna("Unknown")
df_label["Gender_Encoder"] = le.fit_transform(df_label["Gender"]) # 1-Male & 0-Female

#df["Gender"] = le.fit_transform(df["Gender"])

print("After Label Encoding: ")
print(df_label)

