import pandas as pd
import numpy as np
import talib
from binance.client import Client
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# Binance API
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'

# Binance Client objesi
client = Client(api_key, api_secret)

# Kline verilerini almak istediğiniz kripto paranın sembolü ve zaman dilimini belirtin
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_15MINUTE

# Türkiye saati (UTC+3) ile çekmek için saat farkını hesaplayın
saat_farki = timedelta(hours=3)

# Verileri çekmek istediğiniz tarih aralığını belirleyin
start_date = "2024-01-01 00:00:00"

# Verileri çek
klines = client.get_historical_klines(symbol, interval, start_date)

# DataFrame oluştur
df = pd.DataFrame(klines)

# Sütunları adlandır
df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']

# İstenmeyen sütunların listesi
drop_columns = ['close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']

# Bu sütunları DataFrame'den kaldır
df.drop(columns=drop_columns, inplace=True)

# Tür dönüşümleri
df['close'] = df['close'].astype(float)
df['open'] = df['open'].astype(float)
df['high'] = df['high'].astype(float)
df['low'] = df['low'].astype(float)
df['volume'] = df['volume'].astype(float)

# 'timestamp' sütununu tarihe dönüştür ve Türkiye saatine göre ayarla
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') + saat_farki

# 'timestamp' sütununu indeks olarak ayarla
df.set_index('timestamp', inplace=True)

# RSI ve EMA hesaplamaları
df['RSI'] = talib.RSI(df['close'], timeperiod=14)
df['EMA14'] = talib.EMA(df['close'], timeperiod=14)
df['EMA50'] = talib.EMA(df['close'], timeperiod=50)
df['EMA100'] = talib.EMA(df['close'], timeperiod=100)
df['EMA200'] = talib.EMA(df['close'], timeperiod=200)

# Bollinger Bantları
df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)

# MACD
df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)

# Gecikmeli özellikler ekleme
lags = [1, 2, 3]

for lag in lags:
    df[f'close_lag{lag}'] = df['close'].shift(lag)
    df[f'open_lag{lag}'] = df['open'].shift(lag)
    df[f'high_lag{lag}'] = df['high'].shift(lag)
    df[f'low_lag{lag}'] = df['low'].shift(lag)
    df[f'volume_lag{lag}'] = df['volume'].shift(lag)
    df[f'RSI_lag{lag}'] = df['RSI'].shift(lag)
    df[f'EMA14_lag{lag}'] = df['EMA14'].shift(lag)
    df[f'EMA50_lag{lag}'] = df['EMA50'].shift(lag)
    df[f'EMA100_lag{lag}'] = df['EMA100'].shift(lag)
    df[f'EMA200_lag{lag}'] = df['EMA200'].shift(lag)
    df[f'upper_band_lag{lag}'] = df['upper_band'].shift(lag)
    df[f'middle_band_lag{lag}'] = df['middle_band'].shift(lag)
    df[f'lower_band_lag{lag}'] = df['lower_band'].shift(lag)
    df[f'MACD_lag{lag}'] = df['MACD'].shift(lag)
    df[f'MACD_signal_lag{lag}'] = df['MACD_signal'].shift(lag)
    df[f'MACD_hist_lag{lag}'] = df['MACD_hist'].shift(lag)

# Hareketli ortalamalar ekleme
df['SMA_20'] = talib.SMA(df['close'], 20)
df['SMA_50'] = talib.SMA(df['close'], 50)

# Sinyal kurallarının eklenmesi
df['Buy_RSI_EMA'] = np.where((df['RSI'] > 30) & (df['close'] > df['EMA14']), 1, 0)
df['Sell_RSI_EMA'] = np.where((df['RSI'] < 70) & (df['close'] < df['EMA14']), 1, 0)

df['Buy_BB_RSI'] = np.where((df['close'] < df['lower_band']) & (df['RSI'] > 30), 1, 0)
df['Sell_BB_RSI'] = np.where((df['close'] > df['upper_band']) & (df['RSI'] < 70), 1, 0)

df['Buy_MACD_RSI'] = np.where((df['MACD'] > df['MACD_signal']) & (df['RSI'] < 30), 1, 0)
df['Sell_MACD_RSI'] = np.where((df['MACD'] < df['MACD_signal']) & (df['RSI'] > 70), 1, 0)

df['Buy_EMA'] = np.where((df['EMA50'] > df['EMA200']), 1, 0)
df['Sell_EMA'] = np.where((df['EMA50'] < df['EMA200']), 1, 0)

df['Buy_ALL'] = np.where((df['EMA50'] > df['EMA200']) & (df['RSI'] > 30) & (df['RSI'] < 70) & (df['close'] > df['middle_band']) & (df['SMA_20'] > df['SMA_50']), 1, 0)
df['Sell_ALL'] = np.where((df['EMA50'] < df['EMA200']) & (df['RSI'] > 30) & (df['RSI'] < 70) & (df['close'] < df['middle_band']) & (df['SMA_20'] < df['SMA_50']), 1, 0)

# Eksik verileri doldurma (forward fill ve backward fill)
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# Anomalileri tespit etme ve kaldırma
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.float64, np.int64])))
df = df[(z_scores < 3).all(axis=1)]

# Belirli sütunları ölçeklendirme
scaler = MinMaxScaler()
scaled_columns = ['close', 'open', 'high', 'low', 'volume', 'RSI', 'EMA14', 'EMA50', 'EMA100', 'EMA200', 'upper_band', 'middle_band', 'lower_band', 'MACD', 'MACD_signal', 'MACD_hist']
df[scaled_columns] = scaler.fit_transform(df[scaled_columns])

# DataFrame'in ilk birkaç satırını yazdır
print(df.info())

# DataFrame'i CSV dosyasına kaydet
df.to_csv('btc_15m_scaled.csv')
