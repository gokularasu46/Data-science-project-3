import pandas as pd
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'size':[500,800,1000,1200,1500],
    'price':[1500000,2500000,3000000,3500000,4500000]
}

df = pd.DataFrame(data)

X = df[['size']]
y = df['price']

model = LinearRegression()
model.fit(X,y)

# Predict price
price = model.predict([[900]])

print("Predicted House Price:",price)