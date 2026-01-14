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
MODEL_FILE = 'fabio_relative_model.pkl'
MEMORY_FILE = 'fabio_memory.csv'

# TELEGRAM KEYS (PASTE YOURS HERE)
TELEGRAM_TOKEN = "8246165743:AAFHcF8NpJmmDsLAZjRoTm4nZaa3MUT4Z5M" 
TELEGRAM_CHAT_ID = "5291207565"

# 🎯 RISK MANAGEMENT (THE FABIO SETTINGS)
ATR_SL_MULTIPLIER = 1.5   # Stop Loss = 1.5x Volatility (Tight)
RISK_REWARD = 2.5         # Target = 2.5x Risk (Big Wins)
CONFIDENCE_THRESHOLD = 55.0

# ==========================================
# 🔧 SETUP
# ==========================================
print("🔌 Connecting to Binance...")
exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})

print(f"🧠 Loading {MODEL_FILE}...")
try:
    model = joblib.load(MODEL_FILE)
    print("✅ Model Loaded!")
except:
    print(f"❌ Error: {MODEL_FILE} not found! Upload it first.")
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
        print("📥 Initializing Memory (Downloading 1000 candles)...")
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
# 🧮 FEATURE ENGINEERING
# ==========================================
def prepare_features(df):
    if len(df) < 300: return None, None, None

    # 1. Indicators
    window = 288
    df['Typical'] = (df['high'] + df['low'] + df['close']) / 3
    df['Vol_Price'] = df['Typical'] * df['volume']
    
    df['VWAP'] = df['Vol_Price'].rolling(window).sum() / df['volume'].rolling(window).sum()
    df['VWAP_Std'] = df['close'].rolling(window).std()
    
    df['VAH'] = df['VWAP'] + (2.0 * df['VWAP_Std'])
    df['VAL'] = df['VWAP'] - (2.0 * df['VWAP_Std'])

    # 2. Relative Distance
    df['Pct_Above_VAL'] = (df['close'] - df['VAL']) / df['VAL'] * 100
    df['Pct_Below_VAH'] = (df['VAH'] - df['close']) / df['VAH'] * 100
    df['Dist_VWAP'] = (df['close'] - df['VWAP']) / df['VWAP'] * 100

    # 3. Absorption/Momentum
    df['Avg_Vol'] = df['volume'].rolling(20).mean()
    df['Vol_Ratio'] = df['volume'] / df['Avg_Vol']
    df['Body'] = abs(df['close'] - df['open'])
    df['Avg_Body'] = df['Body'].rolling(20).mean()
    
    df['Absorption'] = np.where((df['Vol_Ratio'] > 2.0) & (df['Body'] < df['Avg_Body'] * 0.5), 1, 0)
    df['Momentum'] = np.where((df['Vol_Ratio'] > 2.0) & (df['Body'] > df['Avg_Body'] * 1.5), 1, 0)

    # 4. ATR (For Stop Loss)
    df['tr'] = np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift()))
    df['ATR'] = df['tr'].rolling(14).mean()

    last = df.iloc[-1]
    
    input_data = {
        'Pct_Above_VAL': [last['Pct_Above_VAL']],
        'Pct_Below_VAH': [last['Pct_Below_VAH']],
        'Dist_VWAP': [last['Dist_VWAP']],
        'Vol_Ratio': [last['Vol_Ratio']],
        'Absorption': [last['Absorption']],
        'Momentum': [last['Momentum']]
    }
    
    return pd.DataFrame(input_data), last, last['ATR']

# ==========================================
# 🚀 MAIN LOOP
# ==========================================
print(f"🚀 FABIO BOT STARTED: {SYMBOL}")
print("Mode: Calculating Live Risk (SL/TP)")
send_telegram(f"✅ **Fabio Bot Online**\nRisk Settings: SL={ATR_SL_MULTIPLIER}x ATR, TP={RISK_REWARD}R")

while True:
    try:
        df = update_memory()
        features, candle, atr = prepare_features(df)
        
        if features is not None:
            # Predict
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0]
            confidence = prob.max() * 100
            
            # Live Data
            price = candle['close']
            val = candle['VAL']
            vah = candle['VAH']
            vwap = candle['VWAP']
            
            # --- 🧮 CALCULATE POTENTIAL SL/TP ---
            # Buying Math:
            buy_sl = price - (atr * ATR_SL_MULTIPLIER)
            buy_risk = price - buy_sl
            buy_tp = price + (buy_risk * RISK_REWARD)
            
            # Selling Math:
            sell_sl = price + (atr * ATR_SL_MULTIPLIER)
            sell_risk = sell_sl - price
            sell_tp = price - (sell_risk * RISK_REWARD)

            # --- SIGNAL LOGIC ---
            
            # 1. BUY SIGNAL (Reversion or Momentum)
            if pred == 1:
                # Filter: Must be cheap OR Momentum Breakout
                if (price < val * 1.005) or (candle['Momentum'] == 1 and price > vwap):
                    if confidence > CONFIDENCE_THRESHOLD:
                        msg = (
                            f"🟢 **BUY SIGNAL**\n"
                            f"Price: {price:.2f}\n"
                            f"Conf: {confidence:.1f}%\n"
                            f"----------------\n"
                            f"🛑 STOP LOSS: {buy_sl:.2f}\n"
                            f"🎯 TAKE PROFIT: {buy_tp:.2f}\n"
                            f"Risk: ${buy_risk:.2f} per unit"
                        )
                        print("\n>>> 🟢 BUY SIGNAL SENT")
                        send_telegram(msg)
                        time.sleep(300)
            
            # 2. SELL SIGNAL
            elif pred == 2:
                # Filter: Must be expensive OR Momentum Breakdown
                if (price > vah * 0.995) or (candle['Momentum'] == 1 and price < vwap):
                    if confidence > CONFIDENCE_THRESHOLD:
                        msg = (
                            f"🔴 **SELL SIGNAL**\n"
                            f"Price: {price:.2f}\n"
                            f"Conf: {confidence:.1f}%\n"
                            f"----------------\n"
                            f"🛑 STOP LOSS: {sell_sl:.2f}\n"
                            f"🎯 TAKE PROFIT: {sell_tp:.2f}\n"
                            f"Risk: ${sell_risk:.2f} per unit"
                        )
                        print("\n>>> 🔴 SELL SIGNAL SENT")
                        send_telegram(msg)
                        time.sleep(300)

            # --- DISPLAY DASHBOARD (Every 30s) ---
            print("-" * 50)
            print(f"TIME: {time.strftime('%H:%M:%S')} | PRICE: {price:.2f}")
            print(f"AI: {pred} ({confidence:.0f}%) | ATR: {atr:.2f}")
            print(f"LEVELS: VAL {val:.2f} | VAH {vah:.2f}")
            
            # Show Potential Targets so you can see the math working
            if pred == 1:
                print(f"👀 WATCHING BUY: SL {buy_sl:.2f} | TP {buy_tp:.2f}")
            elif pred == 2:
                print(f"👀 WATCHING SELL: SL {sell_sl:.2f} | TP {sell_tp:.2f}")
            else:
                print(f"💤 WAITING (Neutral)")
            
        time.sleep(30)

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Loop Error: {e}")
        time.sleep(10)