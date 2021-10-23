import numpy as np 
import pandas as pd
import yfinance as yf
from datetime import datetime as dt
import time
import streamlit as st
from streamlit_disqus import st_disqus
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from plotly import graph_objs as go
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


st.title('Top 5 Banking Stock Price Prediction Web Application')

st.image("StockPricePredictionWebAppBanner.jpg")

with st.spinner('Wait for it...'):
  time.sleep(5)
  stocks = ('HDFCBANK.NS', 'SBIN.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'KOTAKBANK.NS')
  selected_stock = st.selectbox("Select Stock for Prediction", stocks)
st.success('Loading Data Done!')

START = st.date_input('Enter the starting date for Prediction', value= dt.strptime('2019-01-01', '%Y-%m-%d'))
TODAY = dt.today().strftime('%Y-%m-%d')
st.info("ARIMA model works best on shorter time span, so try to keep the dataframe short inorder to get reasonable result.")

@st.cache(allow_output_mutation=True)
def load_data(ticker):
  data = yf.download(ticker, START, TODAY)
  data.reset_index(inplace=True)
  data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
  data = data.drop(['Adj Close','Volume'], axis = 1)
  return data

data_load_state = st.markdown("Load state...")
data = load_data(selected_stock)
data_load_state.markdown("Loading data...done!")

st.subheader(f'Data of {selected_stock} from {START} till now: ')
st.write(data)

st.subheader(f"Summary of {selected_stock}'s dataset: ")
st.write(data.describe())

st.subheader(f"{selected_stock}'s Financial Statements:")
Fin_Stat = st.radio("Select any one of this to view",
                    ('Income Statement', 'Balance Sheet', 'Cash Flow'))
Ticker = yf.Ticker(selected_stock)
if Fin_Stat == 'Income Statement':
  IS = Ticker.get_financials()
  IS.columns = IS.columns.strftime('%Y-%m-%d')
  st.markdown(f"Income Statement of {selected_stock}")
  st.write(IS)
elif Fin_Stat == 'Balance Sheet':
  BS = Ticker.get_balancesheet()
  BS.columns = BS.columns.strftime('%Y-%m-%d')
  st.markdown(f"Balance Sheet of {selected_stock}")
  st.write(BS)
elif Fin_Stat == 'Cash Flow':
  CF = Ticker.get_cashflow()
  CF.columns = CF.columns.strftime('%Y-%m-%d')
  st.markdown(f"Cash Flow of {selected_stock}")
  st.write(CF)

MA_1st = st.number_input('Insert the number of days for 1st Moving Average: ', value=34)
MA_2nd = st.number_input('Insert the number of days for 2nd Moving Average: ', value=89)
st.caption("You can choose any of these Fibonacci Sequence number for the two Moving Average calculation (0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233....)")
data['MA_1st'] = data['Close'].rolling(MA_1st).mean()
data['MA_2nd'] = data['Close'].rolling(MA_2nd).mean()

st.subheader(f'{selected_stock} VS {MA_1st} days MA VS {MA_2nd} days MA')
def plot_raw_data():
  fig = go.Figure()
  fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name=f'{selected_stock} Close', line_color='blue'))
  fig.add_trace(go.Scatter(x = data['Date'], y = data['MA_1st'], name=f'{MA_1st} days MA', line_color='green'))
  fig.add_trace(go.Scatter(x = data['Date'], y = data['MA_2nd'], name=f'{MA_2nd} days MA', line_color='red'))
  fig.layout.update(xaxis_rangeslider_visible = True)
  st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

@st.cache(allow_output_mutation=True)
def load_df(ticker):
  df = yf.download(ticker, START, TODAY)
  df = df.drop(df.columns[[0,1,2,4,5]], axis=1)
  df = df.asfreq('d')
  df = df.fillna(method= 'ffill')
  return df
df = load_df(selected_stock)
st.subheader(f"{selected_stock} Dataset for forecasting future Closing price:")
st.write(df)

lags = ar_select_order(df, maxlag=30, old_names=False)
model = AutoReg(df['Close'], lags.ar_lags, old_names=False)
model_fit = model.fit()

train_data = df.iloc[50:int(len(df)*0.8)] # 80% minus first 50
test_data = df.iloc[int(len(df)*0.8):] # Last 20%

sample_days = st.number_input("Manipulate the sampling days to improve the training model for prediction:", value=500)
train_model = AutoReg(df['Close'], sample_days, old_names=False).fit(cov_type="HC0")

start = len(train_data)
end = len(train_data) + len(test_data) - 1

prediction = train_model.predict(start=start, end=end, dynamic = True)
prediction = prediction.to_frame()
prediction = prediction.rename(columns = {0 : 'Prediction'})

prediction_days = st.slider("Insert the number of day's Closing price you want to predict in the fututre:", 50, 100, value = 60)
st.caption("The first 50 days price will be used for validation purpose of the model that you can see in the plot.")
forecast = train_model.predict(start=end, end = end + prediction_days, dynamic = True)
forecast = forecast.to_frame()
forecast = forecast.rename(columns = {0 : 'Forecast'}) 

st.subheader(f"{selected_stock}'s Actual Closing VS Training Data VS Predicted future price")
def plot_raw_data():
  fig = go.Figure()
  fig.add_trace(go.Scatter(x = test_data.index.values, y = test_data['Close'], name = 'Actual Closing', line_color = 'blue'))
  fig.add_trace(go.Scatter(x = prediction.index.values, y = prediction['Prediction'], name = 'Training Data', line_color = 'green'))
  fig.add_trace(go.Scatter(x = forecast.index.values, y = forecast['Forecast'], name = 'Predicted Closing', line_color = 'red'))
  fig.layout.update(xaxis_rangeslider_visible = True)
  st.plotly_chart(fig, use_container_width = True)
plot_raw_data()

st.subheader(f"{selected_stock}'s predicted future price for {prediction_days} days:")
st.write(forecast)

st.markdown('**Static plot for visualization:**')
fig2 = plt.figure(figsize=(22,8))
plt.plot(forecast, color='r', label= 'Forecast',linewidth=2)
plt.plot(prediction, color='g', label= 'Training',linewidth=2)
plt.plot(test_data, color= 'b', label= 'Actual Closing',linewidth=2)
plt.title(f"{selected_stock} Stock forecasted price for next {prediction_days} days")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

st.markdown('**Play this audio file inorder to understand each of the element of the Web Application:**')
st.audio("Voiceover for stock price prediction web app.mp3")

agree = st.checkbox('Check this box to see the warning!')
if agree:
  st.error('''This Dashboard is made for educational and research purpose only. Don't take this
  as an investment advice or don't indulge in investing in any of the aforementioned stocks.''')

st.caption('''Please press the "Comment" button below to give your valuable suggestion.''')
if st.button('Comment'):
  st_disqus("comment-box")
