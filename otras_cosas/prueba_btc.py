import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import requests
import os
import time
from datetime import datetime

# ================================
# 1. Preparación de datos
# ================================
def fetch_market_data(symbol="BTCUSDT", interval="1m", limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore"])
        df = df[["time", "open", "high", "low", "close", "volume"]]
        for col in df.columns:
            df[col] = df[col].astype(float)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error al obtener datos de mercado: {e}")
        return pd.DataFrame()

# ================================
# 2. Modelo de predicción
# ================================
def create_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(50, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model

# ================================
# 3. Lógica de Trading
# ================================
def trading_logic(predicted_price, current_price, capital, position, commission=0.001):
    threshold = 0.002  # 0.2% para activar compra/venta
    if predicted_price > current_price * (1 + threshold):
        if capital > 0:
            position += capital * (1 - commission) / current_price
            capital = 0
            print(f"Comprado a {current_price}")
    elif predicted_price < current_price * (1 - threshold):
        if position > 0:
            capital += position * current_price * (1 - commission)
            position = 0
            print(f"Vendido a {current_price}")
    return capital, position

# ================================
# 4. Reentrenamiento incremental
# ================================
def retrain_model(model, scaler, data, lookback):
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    return model

# ================================
# 5. Ejecución en tiempo real
# ================================
if __name__ == "__main__":
    capital = 1000
    position = 0
    symbol = "BTCUSDT"
    interval = "1m"
    lookback = 60
    model_path = "trading_model.h5"
    scaler = MinMaxScaler(feature_range=(0, 1))

    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = create_model((lookback, 5))  # 5 features: open, high, low, close, volume

    historical_data = []
    trades = []
    start_time = time.time()

    print("Iniciando trading en tiempo real...")

    while True:
        try:
            df = fetch_market_data(symbol=symbol, interval=interval, limit=lookback)
            if df.empty:
                print("Advertencia: Datos de mercado no disponibles, reintentando...")
                time.sleep(10)
                continue

            historical_data.append(df.iloc[-1].values[1:])  # Excluimos el timestamp
            if len(historical_data) > 1000:
                historical_data = historical_data[-1000:]

            scaled_data = scaler.fit_transform(np.array(historical_data))

            X_test = np.array([scaled_data[-lookback:]])

            predicted_price = scaler.inverse_transform(model.predict(X_test))[0, 0]
            current_price = df['close'].iloc[-1]

            error = abs(predicted_price - current_price) / current_price * 100
            print(f"Error: {error:.2f}%")

            prev_capital, prev_position = capital, position
            capital, position = trading_logic(predicted_price, current_price, capital, position)

            if capital != prev_capital or position != prev_position:
                trades.append({
                    'time': datetime.now(),
                    'action': 'BUY' if position > prev_position else 'SELL',
                    'price': current_price,
                    'amount': abs(position - prev_position),
                    'capital': capital,
                    'position': position
                })

            total_value = capital + position * current_price
            profit = total_value - 1000
            roi = (profit / 1000) * 100

            print(f"Predicción: {predicted_price:.2f}, Actual: {current_price:.2f}")
            print(f"Capital: {capital:.2f}, Posición: {position:.4f}")
            print(f"Valor Total: {total_value:.2f}, ROI: {roi:.2f}%")
            print(f"Número de operaciones: {len(trades)}")
            print(f"Tiempo transcurrido: {(time.time() - start_time) / 3600:.2f} horas")

            if len(historical_data) >= lookback and len(historical_data) % 100 == 0:
                print("Reentrenando modelo...")
                model = retrain_model(model, scaler, np.array(historical_data), lookback)
                model.save(model_path)

            time.sleep(60)  # Esperar 1 minuto antes de la próxima iteración

        except Exception as e:
            print(f"Error en el bucle principal: {e}")
            time.sleep(60)
