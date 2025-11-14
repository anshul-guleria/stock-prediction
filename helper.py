from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow.keras as keras
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os


def download_stock_data(stock):
    ticker = stock
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365) 
    df = yf.download(ticker, start=start_date, end=end_date)
    df.to_csv(f"stock_data/{ticker}_last_5_years.csv")
    df=pd.read_csv(f"stock_data/{stock}_last_5_years.csv")
    df=df.drop([0,1]).rename(columns={"Price":"Date"}).reset_index(drop=True)
    df.to_csv(f"stock_data/{stock}_last_5_years.csv", index=False)

def next_7_days_predictions(model, df, scaler):
    target_col='close'
    dataset=df[target_col].values.reshape(-1,1)

    last_60_days=dataset[-60:]
    scaled_last_60_days=scaler.transform(last_60_days)

    future_predictions = []

    window = scaled_last_60_days.reshape(1, 60, 1)

    for _ in range(4):
        predicted_scaled_price = model.predict(window)
        pred=scaler.inverse_transform(predicted_scaled_price)[0][0]
        future_predictions.append(pred)

        #update the window
        new_window = np.append(window[0,1:,0], predicted_scaled_price[0][0])
        window = new_window.reshape(1, 60, 1)

    return future_predictions

def visualize_dataset(stock_df,stock_name):
    # Data visualization for all stocks
    # stock_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    #stock price data
    plt.figure(figsize=(12, 6))
    plt.plot(stock_df['Date'], stock_df['Open'], label='Open', color='blue')
    plt.plot(stock_df['Date'], stock_df['Close'], label='Close', color='red')
    plt.title(f'Open and Close Prices of {stock_name} over time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    if not os.path.exists(f'outputs/{stock_name}'):
        os.makedirs(f'outputs/{stock_name}')
    plt.savefig(f'outputs/{stock_name}/stock_prices.png')
    plt.close()

    #trading volume
    plt.figure(figsize=(12, 6))
    plt.bar(stock_df['Date'], stock_df['Volume'], color='orange')
    plt.title(f'Trading Volume of {stock_name} over time')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    if not os.path.exists(f'outputs/{stock_name}'):
        os.makedirs(f'outputs/{stock_name}')
    plt.savefig(f'outputs/{stock_name}/trading_volume.png')
    plt.close()

    #correlation heatmap
    numeric_data = stock_df.select_dtypes(include=['int64', 'float64'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Matrix for {stock_name}')
    if not os.path.exists(f'outputs/{stock_name}'):
        os.makedirs(f'outputs/{stock_name}')
    plt.savefig(f'outputs/{stock_name}/correlation_matrix.png')
    plt.close()


def retrain_model(stock_name):
    download_stock_data(stock_name)

    df= pd.read_csv(f"stock_data/{stock_name}_last_5_years.csv")
    df['Date']=pd.to_datetime(df['Date'], errors='coerce')

    visualize_dataset(df, stock_name) #plots for the stock

    target_col='Close'

    dataset=df[target_col].values.reshape(-1,1)
    training_data_len=int(np.ceil(len(dataset) * 0.95))
    # print(f"Training data length = {training_data_len}")


    #Preprocessing data (scaling)
    scaler=StandardScaler()
    scaled_data=scaler.fit_transform(dataset)

    #Create training data
    train_data=scaled_data[:training_data_len]

    X_train=[]
    y_train=[]

    #Creating 60 days sliding window for LSTM
    for i in range(60, len(train_data)):
        X_train.append(train_data[i-60:i, 0])   #60 previous days
        y_train.append(train_data[i, 0])        #61st day
    
    #Reshaping data for LSTM input(samples, time_steps, features)
    X_train, y_train=np.array(X_train), np.array(y_train)
    X_train=np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    #Build LSTM model
    model=keras.Sequential()

    #First layer
    model.add(keras.layers.LSTM(units=64,return_sequences=True,input_shape=(X_train.shape[1],1)))

    #Second layer
    model.add(keras.layers.LSTM(units=64,return_sequences=False))

    #Third layer
    model.add(keras.layers.Dense(units=25))

    #Fourth layer
    model.add(keras.layers.Dropout(0.5))

    #Final output layer
    model.add(keras.layers.Dense(units=1))

    # model.summary()

    model.compile(optimizer='adam',loss='mae', metrics=[keras.metrics.RootMeanSquaredError()])

    model.fit(X_train,y_train, epochs=20, batch_size=32)

    model.save(f"models/{stock_name}_model.keras")

    joblib.dump(scaler, f"models/scalers/{stock_name}_scaler.joblib")


    test_data=dataset[training_data_len - 60:]
    scaled_test_data=scaler.transform(test_data)

    #Prepare the test data
    X_test=[]
    y_test=dataset[training_data_len:]

    for i in range(60, len(scaled_test_data)):
        X_test.append(scaled_test_data[i-60:i, 0])


    X_test=np.array(X_test)
    X_test=np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    #Make a predictions
    predictions=model.predict(X_test)
    predictions=scaler.inverse_transform(predictions)

    #Plotting the data and the predictions (saved in outputs folder)
    train=df[:training_data_len]
    valid=df[training_data_len:]

    valid['Predictions']=predictions

    plt.figure(figsize=(16,8))
    plt.title(f"{stock_name} Stock Price Prediction")
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Date'], train['Close'], label='Train Close Price')
    plt.plot(valid['Date'], valid['Close'], label='Actual Close Price')
    plt.plot(valid['Date'], valid['Predictions'], label='Predicted Close Price')
    plt.legend()
    plt.savefig(f"outputs/{stock_name}/stock_price_prediction.png")
    plt.close()

    print(f"Prediction plot for {stock_name} saved in outputs folder.\n")


    return model