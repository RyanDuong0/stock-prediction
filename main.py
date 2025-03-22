import streamlit as st
import pandas as pd
import requests
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go

# Fetch Alpha Vantage API Key from Streamlit secrets
ALPHA_VANTAGE_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data

def fetch_stock_symbols(query):
    """Search for stock symbols using Alpha Vantage API"""
    url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={query}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "bestMatches" in data:
            return data["bestMatches"]
    return []

def plot_raw_data(data):
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    figure.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    figure.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(figure)

st.title("Stock Prediction App")

# Search for stock symbols
search_query = st.text_input("Enter stock name or symbol:")
if search_query:
    stock_matches = fetch_stock_symbols(search_query)
    if stock_matches:
        # Show the stock name and company name
        selected_stock = st.selectbox("Select a stock:", [match["1. symbol"] for match in stock_matches])
        selected_company_name = [match["2. name"] for match in stock_matches if match["1. symbol"] == selected_stock][0]

        # Display Stock Name and Company Name
        st.subheader(f"Stock: {selected_stock} - {selected_company_name}")

    else:
        st.error("No stocks found. Try a different search.")
        st.stop()
else:
    st.warning("Please enter a stock name or symbol.")
    st.stop()

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... done!")

st.subheader('Raw Data')
st.write(data.tail())

plot_raw_data(data)

# Ensure 'Close' is numeric before renaming
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

# Drop any missing values
data = data.dropna(subset=['Close'])

# Convert Date and Close to Prophet format
dataframe_training = data[['Date', 'Close']].copy()
dataframe_training = dataframe_training.rename(columns={"Date": "ds", "Close": "y"}) 

# Ensure 'y' column is numeric
dataframe_training['y'] = pd.to_numeric(dataframe_training['y'], errors='coerce')

if dataframe_training.empty:
    st.error("Error: No data available for modeling.")
    st.stop()

# Train the model
model = Prophet()
model.fit(dataframe_training)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())

st.write('Forecast Plot')
figure_forecast = plot_plotly(model, forecast)
st.plotly_chart(figure_forecast)

st.write('Forecast Components')
fig2 = model.plot_components(forecast)
st.write(fig2)
