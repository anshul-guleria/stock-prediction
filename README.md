# stock-prediction

This program predict the next day stock prices.

To run the model download the repository and run main_app.py file using <code>streamlit run_main.py</code>

## Model used - LSTM
I have trained a deep learning model using LSTM
<code>
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
</code>

## Frontend - Streamlit
Used streamlit to create a dashboard for predictions as well as live(freshly downloaded) dataset.
You can add multiple stocks according to your specifications.

## Outputs
For every stock we have:
<code>
stock_name_last_5_years.csv
</code>
<br/>
And other visual plots:
<code>
outputs/stock_folder/correlation_matrix.png
outputs/stock_folder/stock_price_prediction.png
outputs/stock_folder/stock_prices.png
outputs/stock_folder/trading_volume.png
</code>
</br>
The models are save in <code>models</code> folders which also contains scalers (StandardScaler) used while training the model(for preprocessing step)
