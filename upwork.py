import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet  # or use ARIMA/statsmodels if Prophet not available
from neuralprophet import NeuralProphet
from pprint import pprint

# Load the CSV
df = pd.read_csv("2025-07-02_transaction_report.csv")

# Preprocess (adjust column names as needed)
df['Date'] = pd.to_datetime(df['Date'])
#Transaction Type == "Transaction Type" or 'Fixed-price'

df = df[df['Transaction Type'].isin(['Fixed-price', 'Hourly'])]  # Filter for relevant transaction types
monthly = df.groupby(df['Date'].dt.to_period('M'))
# Group by month

monthly = df.groupby(df['Date'].dt.to_period('M')).sum(numeric_only=True).reset_index()
monthly['Date'] = monthly['Date'].dt.to_timestamp()

# Prepare data for Prophet
prophet_df = monthly.rename(columns={'Date': 'ds', 'Amount $': 'y'})

# Train Prophet model
model = Prophet()
model.fit(prophet_df)

# Forecast
future = model.make_future_dataframe(periods=24, freq='M')
forecast = model.predict(future)

# Plot
model.plot(forecast)
plt.title("Earnings Forecast")
plt.savefig('upwork-forecast.jpg', format='jpg')