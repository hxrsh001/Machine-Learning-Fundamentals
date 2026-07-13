# linear regression
import pandas as pd
from sklearn.linear_model import LinearRegression
 
data = {
    "experience":[1,2,3,4,5],
    "salary":[3,5,7,9,11]
}
df = pd.DataFrame(data)
x = df[['experience']] # input single
y = df['salary']
 
model = LinearRegression()
model.fit(x,y)
 
# prediction
new_data  = pd.DataFrame({
    "experience":[6]
})
pred = model.predict(new_data)
print(pred[0])