import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide")

st.sidebar.title("Stock Price Analysis")
st.sidebar.markdown("Use this dashboard to predict stock prices using a pre-trained LSTM model.")   

option=st.sidebar.selectbox("Select an option", ["Live Analysis", "View Predictions"])

if option=="Live Analysis":
    run_btn=st.sidebar.button("Run")
else:
    predict_btn=st.sidebar.button("Predict")

# title and description
st.title("Stock Price Prediction Dashboard")
st.markdown("This dashboard allows you to upload stock price data and predict future prices using a pre-trained LSTM model.")


col1, col2 = st.columns(2)
run_btn=True
#in case of live analysis
if run_btn:
    with col1:
        stock = st.selectbox("Choose a Stock", ["AAPL","GOOGL","MSFT","AMZN","TSLA"])  
    with col2:
        range_years=st.slider("Select Range of Years", 1, 10, 5)
    analyze = st.button("Run Analysis")
    
    if(analyze):
        with st.spinner("Downloading Data..."):
            ticker = stock
            end_date = datetime.now()
            start_date = end_date - timedelta(days=range_years*365) 
            live_data = yf.download(ticker, start=start_date, end=end_date)
            live_data.reset_index(inplace=True)
            # df.to_csv(f"{ticker}_last_5_years.csv")

        live_data.columns = ["date", "close", "high", "low", "open", "volume"]
        st.success("Data Downloaded Successfully!")

        # print(live_data.info())
        # st.subheader(f"Live Daily Stock Data for {ticker}")
        # st.dataframe(live_data.tail())

        # Last 7 days data table with percentage change
        st.subheader("Last 7 Days Live Data with Percentage Change")
        last_7_days = live_data.tail(7).copy()
        last_7_days['Percentage Change'] = last_7_days['close'].pct_change() * 100
        st.dataframe(last_7_days.style.format({'Percentage Change': '{:.2f}%'}))

        col1, col2 = st.columns(2)

        with col1:
            #select closing price over time
            st.subheader("Closing Price Over Time")
            fig = plt.figure(figsize=(12, 6))
            plt.plot(live_data['date'], live_data['close'], label='Closing Price', color='green')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'Closing Price Over Time for {ticker}')
            plt.legend()
            st.pyplot(fig)

        with col2:
            #moving averages
            st.subheader("Moving Averages")
            ma100 = live_data.close.rolling(100).mean()
            ma200 = live_data.close.rolling(200).mean()
            fig = plt.figure(figsize=(12, 6))
            plt.plot(live_data['date'], live_data['close'], label='Closing Price', color='blue')
            plt.plot(live_data['date'], ma100, label='100-Day MA', color='red')
            plt.plot(live_data['date'], ma200, label='200-Day MA', color='green')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'Moving Averages for {ticker}')
            plt.legend()
            st.pyplot(fig)

        #candlestick chart
        st.subheader("Candlestick Chart(Last 60 Days)")
        plot_data=live_data.set_index('date')
        fig=go.Figure(data=[go.Candlestick(x=plot_data.index,
                        open=plot_data['open'], high=plot_data['high'],
                        low=plot_data['low'],
                        close=plot_data['close'])
                        ])
        # min_val = plot_data['low'].min()
        # max_val = plot_data['high'].max()
        # fig.update_yaxes(range=[min_val * 0.98, max_val * 1.02])
        fig.update_layout(
            height=600
        )
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                range=[plot_data.index[-60], plot_data.index[-1]]
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        

        
