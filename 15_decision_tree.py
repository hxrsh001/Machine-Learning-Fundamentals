# decision tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
 
# data set
data = {
    "age":[20,22,25,28,30,35,40,45,50,55],
    "salary":[20000,25000,27000,32000,45000,50000,60000,70000,80000,9000],
    "buy":[0,0,0,0,1,1,1,1,1,0]
}
 
df = pd.DataFrame(data)
print(df)
 
# features and target
x = df[['age','salary']] # features
y = df['buy'] # target
 
# train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
 
# model
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)
 
# prediction
 
y_pred = model.predict(x_test)

print("actual data",y_test)
print("prediction",y_pred)
 