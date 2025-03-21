import streamlit as st
import pandas as pd
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data

def plot_raw_data(data):
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    figure.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    figure.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(figure)

st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... done!")

st.subheader('Raw Data')
st.write(data.tail())

plot_raw_data(data)

# Debugging Step 1: Print first few rows
st.subheader("Debugging: Check Raw Data Format")
st.write(data[['Date', 'Close']].head())

# Ensure 'Close' is numeric before renaming
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

# Drop any missing values
data = data.dropna(subset=['Close'])

# Convert Date and Close to Prophet format
dataframe_training = data[['Date', 'Close']].copy()
dataframe_training = dataframe_training.rename(columns={"Date": "ds", "Close": "y"}) 

# Debugging Step 2: Check if dataframe_training is empty
if dataframe_training.empty:
    st.error("Error: dataframe_training is empty! No data available for modeling.")
    st.stop()

# Ensure 'y' column is numeric
dataframe_training['y'] = pd.to_numeric(dataframe_training['y'], errors='coerce')

# Debugging Step 3: Print first few rows of dataframe_training
st.subheader("Debugging: Check Training Data Format")
st.write(dataframe_training.head())

# Train the model
model = Prophet()
model.fit(dataframe_training)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('Forecast data')
figure_forecast = plot_plotly(model, forecast)
st.plotly_chart(figure_forecast)

st.write('Forecast components')
fig2 = model.plot_components(forecast)
st.write(fig2)
