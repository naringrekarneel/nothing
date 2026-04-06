import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import kagglehub, glob, os

path = kagglehub.dataset_download("harishkumardatalab/housing-price-prediction")
file = glob.glob(os.path.join(path, "*.csv"))[0]
df = pd.read_csv(file)

df = df.dropna()
df.columns = df.columns.str.lower()

cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning']
for c in cols:
    if c in df.columns:
        df[c] = df[c].map({'yes':1,'no':0})

X = df[['area']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model1 = LinearRegression()
model1.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)

print("Simple Linear Regression")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred1)))
print("R2:", r2_score(y_test, y_pred1))

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred1)
plt.title("Area vs Price")
plt.show()


features = ['area','bedrooms','bathrooms','stories'] + cols
X = df[features]
y = df['price']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model2 = LinearRegression()
model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)

print("\nMultiple Linear Regression")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred2)))
print("R2:", r2_score(y_test, y_pred2))

plt.scatter(y_test, y_pred2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Multiple Regression")
plt.show()