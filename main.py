"""
!pip install numpy
!pip install matplotlib
!pip install pandas-datareader
!pip install tensorflow
!pip install scikit-learn
!pip install yfinance
!pip install streamlit
"""

# Import Libraries

import numpy as np
import pandas as pd

# For plotting
import matplotlib.pyplot as plt

# For processing time
import datetime as dt

# To scale data, as Neural Network works better with scaled data.
from sklearn.preprocessing import MinMaxScaler

# Sequential Neural Network model. 
# Dropout: to prevent overfitting, Dense: ouput layer (only one node), LSTM: Long Short Term Memory 
from tensorflow import keras
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential


# Streamlit to run the code as web app
import streamlit as st


# For reading stock data from yahoo
import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()


# Streamlit App
# Set title
st.set_page_config(page_title="Crypto Currency Price Prediction using an LSTM neural network", layout='wide')
st.title("Crypto Currency Price Prediction using an LSTM neural network")


crypto_currency = st.selectbox("Choose a crypto currency: ", options=['ETH', 'BTC', 'XRP', 'DOGE', 'USDT'])    
against_currency = st.selectbox("Choose a currency: ", options=['EUR', 'USD'])

# Training start and end date
train_start = dt.datetime(2020, 1, 1)
train_end = dt.datetime(2023, 1, 1)

# Testing start and end date
test_start = dt.datetime(2023, 1, 1)
test_end = dt.datetime(2023, 4, 1)

if st.button('Train'):
    # Warn that training might take some time
    st.write('Training Might take some time, Training takes place everytime you click on the Train button.')
    # Download data using yf
    data = yf.download(f'{crypto_currency}-{against_currency}', start=train_start, end=train_end)

    # Prepare Data, do not forget to scale data when inputting to and outputting from the Neural Network
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    # to make scaler accessible from other streamlit buttons
    st.session_state.scaler = scaler


    # How many days to look back
    prediction_days = 60
    future_Day = 30
    # To make them available in other buttons
    st.session_state.prediction_days = prediction_days
    st.session_state.future_Day = future_Day


    x_train, y_train = [], []

    for x in range(prediction_days, len(scaled_data)-future_Day):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x+future_Day, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


    # Create Neural Network
    # pip install numpy=1.19.5 if run into issues

    model = Sequential()

    model.add(LSTM(units=30, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=30, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=30))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # to make model available in other buttons
    st.session_state.model = model
    

    # Donwload testing data
    test_data = yf.download(f'{crypto_currency}-{against_currency}', test_start, test_end)

    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)


    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)


    # Accuracy
    mape = np.mean(np.abs(np.array(prediction_prices) - np.array(actual_prices))/np.abs(actual_prices))
    st.text(f'Accuracy (MAPE) on the test data: {mape}')

    # Test Diagram on streamlit
    # Plot the actual and predicted prices
    # figsize to make it easier to read
    fig, axs = plt.subplots(2, 1, figsize=(15,15))
    ax = axs[0]
    ax.plot(actual_prices, color='black', label='Actual Prices')
    ax.plot(prediction_prices, color='green', label='Predicted Prices')
    ax.set_title(f'{crypto_currency} Price Prediction Test')
    ax.set_xlabel(f'Day range from {test_start.strftime("%B %d, %Y")} to {test_end.strftime("%B %d, %Y")}')
    ax.set_ylabel(f'Price in {against_currency}')
    ax.legend(loc='upper left')

    fig.tight_layout(pad=5.0)
    # Plot Training Graph
    st.subheader('Testing Graph')
    st.pyplot(fig=fig)

    # So that we can plot both test and predict data
    st.session_state.fig = fig
    st.session_state.axs = axs

if st.button('Predict'):
    scaler = st.session_state.scaler
    model = st.session_state.model
    prediction_days = st.session_state.prediction_days
    future_Day = st.session_state.future_Day


    # Predict next 30 days
    last_end = dt.datetime.now()
    last_start = dt.datetime.now() - dt.timedelta(60)


    # download data for last 60 days
    last_data = yf.download(f'{crypto_currency}-{against_currency}', last_start, last_end)
    last_data = last_data['Close'].values


    data = []
    for i in range(future_Day):
        real_data = last_data[len(last_data) - prediction_days + i: len(last_data) - prediction_days + future_Day + i]
        real_data = real_data.reshape(-1, 1)
        real_data = scaler.transform(real_data)
        data.append(real_data)

    data = np.array(data)
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))


    # store predicted prices for next 30 days
    predictions = model.predict(data)
    predictions = scaler.inverse_transform(predictions)


    # Convert output to 1D array for plotting purposes
    predictions = np.array(predictions).ravel()

    # Plot the predicted prices
    axs = st.session_state.axs
    fig = st.session_state.fig
    ax2 = axs[1]


    ax2.plot(predictions, color='green', label=f'Predicted prices for the Next {future_Day} days')
    ax2.set_title(f'{crypto_currency} Price Prediction')
    ax2.set_xlabel(f'Next Days from {last_end.strftime("%B %d, %Y")}')
    ax2.set_ylabel(f'Price in {against_currency}')
    ax2.legend(loc='upper left')


    # Plot Prediction Graph
    st.subheader('Prediction Graph')
    st.pyplot(fig=fig)








