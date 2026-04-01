# Description: Short example for Financial Engineering and Time Series Analytics.



from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import logging
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



df = pd.DataFrame({'price': [100, 102, 105, 107, 110, 115]})
df['lag_1'] = df['price'].shift(1)
df['diff'] = df['price'].diff()
logger.info(df)

df['returns'] = df['price'].pct_change()
df['volatility'] = df['returns'].rolling(window=3).std()
logger.info(df)

model = ARIMA(df['price'], order=(1, 1, 1))
model_fit = model.fit()
logger.info(model_fit.summary())


X_train = np.random.rand(100, 10, 1)  # Simulated time series data
y_train = np.random.rand(100, 1)
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(10, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=10)
