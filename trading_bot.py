import ccxt
import pandas as pd
import numpy as np
import joblib
import time
import os
import requests
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
SYMBOL = 'ETH/USDT'
TIMEFRAME = '5m'
LEVERAGE = 30
MODEL_FILE = 'regime_model_v1.pkl'
MEMORY_FILE = 'market_memory.csv'

# TELEGRAM

TELEGRAM_TOKEN = "8246165743:AAFHcF8NpJmmDsLAZjRoTm4nZaa3MUT4Z5M" 
TELEGRAM_CHAT_ID = "5291207565"

# ADVANCED RISK SETTINGS
ATR_SL_MULTIPLIER = 1.5 
RISK_REWARD = 2.0
CONFIDENCE_THRESHOLD = 50.0 

# ==========================================
# 🔧 SETUP
# ==========================================
print("🔌 Connecting to Binance...")
exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})

print(f"🧠 Loading {MODEL_FILE}...")
try:
    model = joblib.load(MODEL_FILE)
    print("✅ Regime Model Loaded!")
except:
    print(f"❌ Error: {MODEL_FILE} not found!")
    exit()

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
        requests.post(url, data=data, timeout=5)
    except:
        pass

# ==========================================
# 💾 MEMORY SYSTEM
# ==========================================
def update_memory():
    if not os.path.exists(MEMORY_FILE):
        print("📥 Downloading 1000 candles history...")
        bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=1000)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.to_csv(MEMORY_FILE, index=False)
        return df
    
    df_old = pd.read_csv(MEMORY_FILE)
    last_time = df_old['timestamp'].iloc[-1]
    new_bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=100)
    df_new = pd.DataFrame(new_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_new = df_new[df_new['timestamp'] > last_time]
    
    if not df_new.empty:
        df_updated = pd.concat([df_old, df_new], ignore_index=True).tail(2000)
        df_updated.to_csv(MEMORY_FILE, index=False)
        return df_updated
    return df_old

# ==========================================
# 🧮 ADVANCED FEATURE ENGINEERING
# ==========================================
def prepare_features(df):
    if len(df) < 300: return None, None, None

    # 1. ADX (Trend Strength)
    period = 14
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['close'].shift(1))
    df['tr2'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df['atr'] = df['tr'].rolling(period).mean()
    
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    df['plus_di'] = 100 * (df['plus_dm'].rolling(period).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(period).mean() / df['atr'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(period).mean()

    # 2. EMA Trend
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['Trend_Score'] = np.where((df['close'] > df['ema_50']) & (df['ema_50'] > df['ema_200']), 1, 
                        np.where((df['close'] < df['ema_50']) & (df['ema_50'] < df['ema_200']), -1, 0))

    # 3. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 4. Volatility Squeeze
    df['std'] = df['close'].rolling(20).std()
    df['bb_width'] = (4 * df['std']) / df['close']
    df['vol_squeeze'] = np.where(df['bb_width'] < df['bb_width'].rolling(50).min() * 1.2, 1, 0)

    # 5. ATR for Risk
    df['ATR_Risk'] = df['tr'].rolling(14).mean()

    # Clean & Select
    df_clean = df.dropna().tail(1)
    if df_clean.empty: return None, None, None

    features = ['adx', 'plus_di', 'minus_di', 'Trend_Score', 'RSI', 'vol_squeeze', 'volume']
    return df_clean[features], df_clean.iloc[0], df_clean['ATR_Risk'].iloc[0]

# ==========================================
# 🚀 MAIN LOOP
# ==========================================
print(f"🚀 HEDGE FUND BOT STARTED: {SYMBOL}")
send_telegram(f"✅ **Advanced Bot Online**\nMode: Regime Filtering (Trend/Range)")

while True:
    try:
        df = update_memory()
        features, candle, atr = prepare_features(df)
        
        if features is not None:
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0]
            confidence = prob.max() * 100
            
            # Market State
            price = candle['close']
            rsi = candle['RSI']
            adx = candle['adx']
            trend_score = candle['Trend_Score']
            
            # DETERMINE REGIME
            regime = "RANGE"
            if adx > 25: regime = "TRENDING"
            
            # --- TRADING LOGIC ---
            
            # 🟢 BUY LOGIC
            if pred == 1: 
                trade_valid = False
                reason = ""
                
                # Case A: Strong Uptrend (Buy the strength)
                if regime == "TRENDING" and trend_score == 1:
                    trade_valid = True
                    reason = "Trend Follow (Strong ADX)"
                
                # Case B: Range Reversal (Buy the dip)
                elif regime == "RANGE" and rsi < 40:
                    trade_valid = True
                    reason = "Range Reversal (Oversold)"
                
                # Execution
                if trade_valid and confidence > CONFIDENCE_THRESHOLD:
                    sl = price - (atr * ATR_SL_MULTIPLIER)
                    tp = price + ((price - sl) * RISK_REWARD)
                    risk_amt = price - sl
                    
                    msg = (f"🟢 **BUY SIGNAL**\nRegime: {regime}\nReason: {reason}\n"
                           f"Price: {price:.2f}\nConf: {confidence:.1f}%\n"
                           f"SL: {sl:.2f} | TP: {tp:.2f}")
                    print(">>> 🔥 BUY SIGNAL SENT")
                    send_telegram(msg)
                    time.sleep(300)
                else:
                    print(f"⚠️ Buy Ignored: Regime={regime}, Trend={trend_score}, RSI={rsi:.1f}")

            # 🔴 SELL LOGIC
            elif pred == 2:
                trade_valid = False
                reason = ""
                
                # Case A: Strong Downtrend (Sell the strength)
                if regime == "TRENDING" and trend_score == -1:
                    trade_valid = True
                    reason = "Trend Follow (Strong ADX)"
                
                # Case B: Range Reversal (Sell the top)
                elif regime == "RANGE" and rsi > 60:
                    trade_valid = True
                    reason = "Range Reversal (Overbought)"
                
                # Execution
                if trade_valid and confidence > CONFIDENCE_THRESHOLD:
                    sl = price + (atr * ATR_SL_MULTIPLIER)
                    tp = price - ((sl - price) * RISK_REWARD)
                    
                    msg = (f"🔴 **SELL SIGNAL**\nRegime: {regime}\nReason: {reason}\n"
                           f"Price: {price:.2f}\nConf: {confidence:.1f}%\n"
                           f"SL: {sl:.2f} | TP: {tp:.2f}")
                    print(">>> 🔥 SELL SIGNAL SENT")
                    send_telegram(msg)
                    time.sleep(300)
                else:
                    print(f"⚠️ Sell Ignored: Regime={regime}, Trend={trend_score}, RSI={rsi:.1f}")
            
            else:
                print(f"Price: {price:.2f} | Regime: {regime} (ADX {adx:.1f}) | AI: {pred} ({confidence:.0f}%)")

        time.sleep(15)

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Loop Error: {e}")
        time.sleep(10)