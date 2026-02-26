import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("=== FINANCE DATA MODEL STARTED ===")

# ------------------------------------------------
# 1. Download Stock Data
# ------------------------------------------------

ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2024-01-01")

# Flatten MultiIndex columns if necessary
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

print("\nColumns:")
print(data.columns)

print("\nFirst 5 Rows:")
print(data.head())

# ------------------------------------------------
# 2. Calculate Daily Returns
# ------------------------------------------------

data['Daily_Return'] = data['Close'].pct_change()

data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))

data = data.dropna()

print("\nReturn Statistics:")
print(data['Daily_Return'].describe())

# ------------------------------------------------
# 3. Volatility (Rolling Std)
# ------------------------------------------------

data['Volatility'] = data['Daily_Return'].rolling(window=30).std()

# ------------------------------------------------
# 4. Visualization
# ------------------------------------------------

plt.figure(figsize=(10,5))
plt.plot(data['Close'])
plt.title("Closing Price")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(data['Volatility'])
plt.title("30-Day Rolling Volatility")
plt.show()

# ------------------------------------------------
# 5. Correlation Matrix
# ------------------------------------------------

plt.figure(figsize=(6,5))
sns.heatmap(data[['Daily_Return','Log_Return','Volatility']].corr(),
            annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# ------------------------------------------------
# 6. Predictive Model
# ------------------------------------------------

data['Next_Day_Return'] = data['Daily_Return'].shift(-1)
data = data.dropna()

X = data[['Daily_Return']]
y = data['Next_Day_Return']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print("\nModel Mean Squared Error:", mse)
print("Model Coefficient:", model.coef_[0])

print("=== FINANCE DATA MODEL COMPLETED ===")
