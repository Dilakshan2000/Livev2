import ccxt
import pandas as pd
import numpy as np
import joblib
import time
import requests  # <--- NEW: For Telegram
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
SYMBOL = 'ETH/USDT'
TIMEFRAME = '15m'
LEVERAGE = 30
MODEL_FILE = 'smc_perfect_model.pkl'

# --- TELEGRAM KEYS (PASTE YOURS HERE) ---
# Keep the quotes "" around the numbers/text


TELEGRAM_TOKEN = "8246165743:AAFHcF8NpJmmDsLAZjRoTm4nZaa3MUT4Z5M" 
TELEGRAM_CHAT_ID = "5291207565"


# RISK SETTINGS
ATR_SL_MULTIPLIER = 2.0  # Stop Loss width
RISK_REWARD = 2.0        # Target Profit multiplier
CONFIDENCE_THRESHOLD = 60.0 # Only trade if AI is >60% sure

# ==========================================
# 🔧 SETUP
# ==========================================
print("🔌 Connecting to Binance Futures...")
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

print("🧠 Loading SMC Perfect Model...")
try:
    model = joblib.load(MODEL_FILE)
    print("✅ Model Loaded!")
except Exception as e:
    print(f"❌ Error: Could not find {MODEL_FILE}")
    exit()

# ==========================================
# 📱 TELEGRAM FUNCTION
# ==========================================
def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID, 
            "text": message,
            "parse_mode": "Markdown" # Makes bold text look nice
        }
        requests.post(url, data=data)
        print("✅ Telegram Sent!")
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

# Test the connection immediately on startup
print("📞 Testing Telegram Connection...")
send_telegram(f"✅ **BOT STARTED**\nSymbol: {SYMBOL}\nStrategy: SMC Perfect Model")

# ==========================================
# 🧮 MATH FUNCTIONS
# ==========================================
def get_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).rolling(window=period).mean()

# ==========================================
# 📊 FEATURE ENGINEERING (Matches Training)
# ==========================================
def prepare_smc_features(df):
    # Ensure we have enough data
    if len(df) < 25: return None, None, None

    # Calculate ATR for Risk Management
    df['ATR'] = get_atr(df, 14)
    current_atr = df['ATR'].iloc[-1]
    
    # Flatten Data (Last 20 candles for the model)
    last_20 = df.tail(20).reset_index(drop=True)
    
    # 1. Create the base row
    input_data = {}
    for i in range(20):
        row = last_20.iloc[i]
        s = str(i + 1)
        input_data[f'open{s}'] = row['open']
        input_data[f'high{s}'] = row['high']
        input_data[f'low{s}'] = row['low']
        input_data[f'close{s}'] = row['close']
        input_data[f'volume{s}'] = row['volume']
    
    row_df = pd.DataFrame([input_data])

    # 2. CALCULATE SMC FEATURES (The "Perfect" Logic)
    
    # FVG Detection
    row_df['Has_Bullish_FVG'] = np.where(
        (row_df['low20'] > row_df['high18']) & (row_df['close19'] > row_df['open19']), 1, 0
    )
    row_df['Has_Bearish_FVG'] = np.where(
        (row_df['high20'] < row_df['low18']) & (row_df['close19'] < row_df['open19']), 1, 0
    )

    # BOS (Break of Structure) Logic
    high_cols = [f'high{i}' for i in range(1, 20)]
    low_cols = [f'low{i}' for i in range(1, 20)]
    
    prev_high_max = row_df[high_cols].max(axis=1)
    prev_low_min = row_df[low_cols].min(axis=1)
    
    row_df['BOS_Bullish'] = np.where(row_df['close20'] > prev_high_max, 1, 0)
    row_df['BOS_Bearish'] = np.where(row_df['close20'] < prev_low_min, 1, 0)

    # 3. Filter Columns (Keep ONLY what the model was trained on)
    feature_cols = ['Has_Bullish_FVG', 'Has_Bearish_FVG', 'BOS_Bullish', 'BOS_Bearish',
                    'open16','close16','open17','close17','open18','close18','open19','close19','open20','close20',
                    'volume20']
    
    try:
        final_features = row_df[feature_cols]
        return final_features, last_20.iloc[-1], current_atr
    except KeyError as e:
        print(f"Column Error: {e}")
        return None, None, None

# ==========================================
# 🚀 MAIN LOOP
# ==========================================
print(f"🚀 SMC-PERFECT BOT STARTED: {SYMBOL}")
print("Waiting for signals...")

while True:
    try:
        # Get Live Data
        bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=100)
        df_full = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Prepare Features
        features, candle, atr = prepare_smc_features(df_full)
        
        if features is not None:
            # Predict
            prediction = model.predict(features)[0]      # 0, 1, or 2
            probabilities = model.predict_proba(features)[0]
            confidence = probabilities.max() * 100
            
            price = candle['close']
            
            # Print Status
            status = "HOLD"
            if prediction == 1: status = "BUY"
            if prediction == 2: status = "SELL"
            print(f"Price: {price:.2f} | Signal: {status} | Conf: {confidence:.1f}%")

            # --- BUY SIGNAL ---
            if prediction == 1 and confidence >= CONFIDENCE_THRESHOLD:
                sl = price - (atr * ATR_SL_MULTIPLIER)
                risk = price - sl
                tp = price + (risk * RISK_REWARD)
                
                msg = (
                    f"🚀 **BUY SIGNAL**\n"
                    f"Pair: {SYMBOL}\n"
                    f"Confidence: {confidence:.1f}%\n"
                    f"----------------\n"
                    f"ENTRY: ${price:.2f}\n"
                    f"STOP LOSS: ${sl:.2f}\n"
                    f"TAKE PROFIT: ${tp:.2f}\n"
                    f"Risk: {RISK_REWARD}R"
                )
                print(">>> 🟢 SENDING BUY ALERT")
                send_telegram(msg)
                time.sleep(300) # Wait 5 mins

            # --- SELL SIGNAL ---
            elif prediction == 2 and confidence >= CONFIDENCE_THRESHOLD:
                sl = price + (atr * ATR_SL_MULTIPLIER) # SL above for short
                risk = sl - price
                tp = price - (risk * RISK_REWARD)      # TP below for short
                
                msg = (
                    f"🔻 **SELL SIGNAL**\n"
                    f"Pair: {SYMBOL}\n"
                    f"Confidence: {confidence:.1f}%\n"
                    f"----------------\n"
                    f"ENTRY: ${price:.2f}\n"
                    f"STOP LOSS: ${sl:.2f}\n"
                    f"TAKE PROFIT: ${tp:.2f}\n"
                    f"Risk: {RISK_REWARD}R"
                )
                print(">>> 🔴 SENDING SELL ALERT")
                send_telegram(msg)
                time.sleep(300)

        time.sleep(15) # Check every 15 seconds
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Loop Error: {e}")
        time.sleep(5)