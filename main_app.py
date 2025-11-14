import tensorflow as tf
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from helper import download_stock_data, next_7_days_predictions, retrain_model,download_stock_data


st.set_page_config(page_title="Stock Prediction Dashboard", page_icon=":bar_chart:", layout="wide")

st.sidebar.title("Stock Price Analysis")
st.sidebar.markdown("Use this dashboard to predict stock prices using a pre-trained LSTM model.")   

option=st.sidebar.selectbox("Select an option", ["Live Analysis", "View Predictions"])

# Initialize states
if "run_btn" not in st.session_state:
    st.session_state.run_btn = False
if "predict_btn" not in st.session_state:
    st.session_state.predict_btn = False

# Use states
run_btn = st.session_state.run_btn
predict_btn = st.session_state.predict_btn

if option=="Live Analysis":
    st.session_state.predict_btn=False
    if st.sidebar.button("Run"):
        st.session_state.run_btn=True
else:
    st.session_state.run_btn=False
    if st.sidebar.button("Predict"):
        st.session_state.predict_btn=True

# title and description
st.title("Stock Price Prediction Dashboard")
st.markdown("This dashboard allows you to upload stock price data and predict future prices using a pre-trained LSTM model.")


col1, col2 = st.columns(2)
#in case of live analysis
# run_btn=True

#in case of viewing predictions
# predict_btn=True

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

elif predict_btn:
    #incase of viewing predictions
    stock_to_predict=st.selectbox("Select Stock for Prediction", ["AAPL","GOOGL","MSFT","AMZN","TSLA","Upload CSV File"])

    @st.dialog("Please Confirm")
    def update_data_dialog():
        st.write("Confirm if you want to update the stock prediction model to today's date data.")
    
        if st.button("Confirm"):
            st.session_state["is_confirmed"] = True
            st.rerun() # Rerun to update the main app after dialog closes

    if "is_confirmed" not in st.session_state:
        st.session_state["is_confirmed"] = False

    if st.button("Update Model with Latest Data"):
        update_data_dialog()
    
    if st.session_state["is_confirmed"]:
        st.session_state["is_confirmed"] = False  # Reset confirmation state
        print("User confirmed to update the model.")
        with st.spinner("Updating Model..."):
            retrain_model(stock_to_predict)
            # download_stock_data(stock_to_predict)
        st.success("Model updated with latest data!")
    
    #--------------------------------------------
    if stock_to_predict=="Upload CSV File":
        #This part is to be developed
        pass
    else:
        model=tf.keras.models.load_model(f"models/{stock_to_predict}_model.keras")
        df=pd.read_csv(f"stock_data/{stock_to_predict}_last_5_years.csv")
        df.columns = ["date", "close", "high", "low", "open", "volume"]
        scaler=joblib.load(f"models/scalers/{stock_to_predict}_scaler.joblib")
        
        predictions=next_7_days_predictions(model, df, scaler)

        custom_css=f"""
        <style>
        .container-next {{
                border: 2px solid #4CAF50;
                background-color: lightgreen;
                border-radius: 5px;
                padding: 10px;
                min-height: 100px;
                width: 80%;
                margin:10px auto;
                font-size: 20px;
        }}

        .container-today {{
                border: 2px solid #FFAC1C;
                background-color: #FFC35C;
                border-radius: 5px;
                padding: 10px;
                min-height: 100px;
                width: 80%;
                margin:10px auto;
                font-size: 20px;
        }}

        .inner-data {{
                font-size: 30px;
                font-weight: bold;
                color: #333;
                margin: auto;
                text-align: center;
        }}
        </style>
        """

        st.markdown(custom_css, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="container-today">Today\'s Stock Price for <b>{stock_to_predict}</b>: ({df['date'].iloc[-1]}) <div class="inner-data">${df["close"].iloc[-1]:.2f}</div></div>', unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="container-next">Predicted Stock Price for <b>{stock_to_predict}</b> (Next Day): <div class="inner-data">${predictions[0]:.2f}</div></div>', unsafe_allow_html=True) 
        
        print(predictions)

        st.header(f"Data Visualizations and Model Testing for {stock_to_predict}")
        st.subheader(f"Stock Data for {stock_to_predict}(Last 7 days)")
        last_7_days = df.tail(7).copy()
        last_7_days['Percentage Change'] = last_7_days['close'].pct_change() * 100
        st.dataframe(last_7_days.style.format({'Percentage Change': '{:.2f}%'}))

        st.header("Predictions vs Actual Prices on Test Data")
        st.image(f"outputs/{stock_to_predict}/stock_price_prediction.png")


        col1, col2 = st.columns(2)
        with col1:
            #closing price over time
            st.subheader("Closing Price Over Time")
            st.image(f"outputs/{stock_to_predict}/stock_prices.png")
        with col2:
            #moving averages
            pass
        with col2:
            st.subheader("Trading Volume Over Time")
            st.image(f"outputs/{stock_to_predict}/trading_volume.png")
        with col1:
            st.subheader("Correlation Heatmap")
            st.image(f"outputs/{stock_to_predict}/correlation_matrix.png")
            
