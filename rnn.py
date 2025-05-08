import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Load Google stock price data
df = pd.read_csv('/content/GOOG.csv')  # Replace with actual path if needed
print(df.head())  # Optional: View the first few rows

# 2. Use only the 'Close' prices
data = df[['Close']].values

# 3. Normalize the data to [0, 1]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 4. Create sequences of 60 time steps
sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# 5. Reshape X to (samples, time_steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 6. Split into training and testing sets (80% train, 20% test)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 7. Build LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='tanh', input_shape=(X.shape[1], 1), return_sequences=True))
model.add(LSTM(units=50, activation='tanh'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# 8. Train the model with more epochs (e.g., 50 epochs)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 9. Predict and inverse transform
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# 10. Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(real_prices, color='blue', label='Actual Google Stock Price')
plt.plot(predicted_prices, color='red', label='Predicted Stock Price')
plt.title('Google Stock Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.tight_layout()
plt.show()

# 11. Numerical results
mse = mean_squared_error(real_prices, predicted_prices)
rmse = np.sqrt(mse)
mae = mean_absolute_error(real_prices, predicted_prices)

print(f"\nEvaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# 12. Save the trained model for future use
model.save('google_stock_price_lstm.h5')
print("\nModel saved as 'google_stock_price_lstm.h5'")
