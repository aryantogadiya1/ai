import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import Ridge
from tensorflow import keras



st.header("Stock market prediction")

st.markdown("##### choose a company from given list")

selected_option = st.selectbox('',
                               ["select company", 'indus bank', 'HDFC bank', 'icici bank', 'kotak mahindra bank'])

all = {
    "indus bank": "INDUSINDBK.NS",
    "HDFC bank": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "kotak mahindra bank": "KOTAKBANK.NS",
}

if selected_option != 'select company':
    ticker = yf.Ticker(all[selected_option])
    data = ticker.history("5mo")
    print(len(data))
    train_x = data['Open']
    train_y = data['High']
    st.write(data)
    a = pd.DataFrame({
        "Open": data['Open'],
        "High": data['High']
    })
    st.line_chart(data)
    print(len(data))
    st.line_chart(a)

    model = keras.Sequential([
        keras.layers.Dense(units=25, activation="relu"),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=1, activation='relu')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(train_x, train_y, epochs=100)

    y = model.predict(train_x)
    print("predicted y ", y)
    print('actual y ', train_y)
    st.write(len(train_y))

    st.markdown("### according to my knoweldge tommorow BEST price of {0} stock is {1}".format(selected_option, y[len(data)-1]))
