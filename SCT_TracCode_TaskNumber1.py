import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Boston housing dataset
boston = fetch_openml(name='boston', version=1, as_frame=True)
df = boston.frame

# Rename columns to lowercase for your analysis
df.rename(columns={
    'RM': 'rooms',
    'LSTAT': 'lstat',
    'PTRATIO': 'ptratio',
    'MEDV': 'price'
}, inplace=True)

# Confirm available columns (for debug)
print("Available columns:", df.columns.tolist())

# ========== First OLS: price ~ rooms ==========
X = df['rooms']
y = df['price']

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()

df['predicted_price'] = model.predict(X_const)

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(x='rooms', y='price', data=df, label='Actual Prices')
sns.lineplot(x='rooms', y='predicted_price', data=df, color='red', label='Regression Line')
plt.title('Linear Regression: Price vs. Rooms')
plt.xlabel('Rooms')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# ========== OLS with Multiple Predictors ==========
X = df[['rooms', 'lstat', 'ptratio']]
y = df['price']

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
print(model.summary())

# Plot residuals
residuals = model.resid
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.xlabel("Residual")
plt.grid(True)
plt.show()

# QQ plot
sm.qqplot(residuals, line='s')
plt.title("Q-Q Plot of Residuals")
plt.show()

# ========== Linear Regression using scikit-learn ==========
model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)

print("MSE:", mean_squared_error(y, pred))
print("R²:", r2_score(y, pred))
