import ccxt
import pandas as pd
import numpy as np
import joblib
import time
import requests
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ CONFIGURATION & KEYS
# ==========================================
SYMBOL = 'ETH/USDT'
TIMEFRAME = '15m'
MODEL_FILE = 'smc_perfect_model.pkl'


TELEGRAM_TOKEN = "" 
TELEGRAM_CHAT_ID = ""


# --- 🧠 SMC RISK MANAGEMENT SETTINGS ---
# Instead of fixed numbers, we use Structure Lookback
SWING_LOOKBACK = 10       # Look back 10 candles to find Swing High/Low for SL
SL_BUFFER_ATR = 0.5       # Add 0.5x ATR buffer to SL so wicks don't stop us out
TARGET_RISK_REWARD = 2.5  # We want 1:2.5 Risk to Reward (SMC usually aims high)
MAX_SL_PERCENT = 0.015    # (1.5%) If structural SL is wider than this, SKIP trade (Too risky)
CONFIDENCE_THRESHOLD = 65.0 # Increased confidence requirement

# ==========================================
# 🔧 SETUP
# ==========================================
print("🔌 Connecting to Binance Futures...")
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

print(f"🧠 Loading {MODEL_FILE}...")
try:
    model = joblib.load(MODEL_FILE)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error: Could not find {MODEL_FILE}. Make sure to upload it!")
    exit()

def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

# ==========================================
# 🧮 ADVANCED CALCULATIONS
# ==========================================
def get_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).rolling(window=period).mean()

def prepare_features(df):
    if len(df) < 30: return None, None, None
    
    # 1. Engineering Features
    df['ATR'] = get_atr(df, 14)
    current_atr = df['ATR'].iloc[-1]
    
    last_20 = df.tail(20).reset_index(drop=True)
    
    # Flatten last 20 candles into one row
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

    # SMC Logic Recreation
    # FVG
    row_df['Has_Bullish_FVG'] = np.where(
        (row_df['low20'] > row_df['high18']) & (row_df['close19'] > row_df['open19']), 1, 0
    )
    row_df['Has_Bearish_FVG'] = np.where(
        (row_df['high20'] < row_df['low18']) & (row_df['close19'] < row_df['open19']), 1, 0
    )
    # BOS
    high_cols = [f'high{i}' for i in range(1, 20)]
    low_cols = [f'low{i}' for i in range(1, 20)]
    row_df['BOS_Bullish'] = np.where(row_df['close20'] > row_df[high_cols].max(axis=1), 1, 0)
    row_df['BOS_Bearish'] = np.where(row_df['close20'] < row_df[low_cols].min(axis=1), 1, 0)

    # Select Columns
    feature_cols = ['Has_Bullish_FVG', 'Has_Bearish_FVG', 'BOS_Bullish', 'BOS_Bearish',
                    'open16','close16','open17','close17','open18','close18','open19','close19','open20','close20',
                    'volume20']
    
    return row_df[feature_cols], df, current_atr

# ==========================================
# 💎 SMART MONEY LEVEL CALCULATOR
# ==========================================
def calculate_smart_levels(direction, current_price, df_full, atr):
    """
    Calculates Stop Loss based on recent Swing High/Low (Structure)
    Calculates Take Profit based on Risk:Reward ratio.
    """
    # Look at the last N candles for structure
    recent_data = df_full.tail(SWING_LOOKBACK + 1) # +1 to include current
    
    if direction == "BUY":
        # SL is below the lowest low of recent structure - Buffer
        lowest_low = recent_data['low'].min()
        stop_loss = lowest_low - (atr * SL_BUFFER_ATR)
        
        # Risk Calculation
        risk_per_share = current_price - stop_loss
        
        # Take Profit (Risk * Reward Ratio)
        take_profit = current_price + (risk_per_share * TARGET_RISK_REWARD)
        
    elif direction == "SELL":
        # SL is above the highest high of recent structure + Buffer
        highest_high = recent_data['high'].max()
        stop_loss = highest_high + (atr * SL_BUFFER_ATR)
        
        # Risk Calculation
        risk_per_share = stop_loss - current_price
        
        # Take Profit
        take_profit = current_price - (risk_per_share * TARGET_RISK_REWARD)
        
    return stop_loss, take_profit

# ==========================================
# 🚀 MAIN LOOP
# ==========================================
print(f"🚀 SMC-PERFECT BOT V2 STARTED: {SYMBOL}")
print(f"Strategy: Structure Based SL | {TARGET_RISK_REWARD}R Targets")
send_telegram(f"✅ **BOT RESTARTED**\nTargeting {TARGET_RISK_REWARD}R Trades on {SYMBOL}")

while True:
    try:
        # 1. Fetch Data
        bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=100)
        df_full = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 2. Get AI Prediction
        features, data_history, atr = prepare_features(df_full)
        
        if features is not None:
            prediction = model.predict(features)[0]
            probs = model.predict_proba(features)[0]
            confidence = probs.max() * 100
            
            current_price = df_full['close'].iloc[-1]

            # 3. Decision Logic
            if confidence >= CONFIDENCE_THRESHOLD:
                
                # --- BUY LOGIC ---
                if prediction == 1:
                    sl, tp = calculate_smart_levels("BUY", current_price, df_full, atr)
                    risk_pct = (current_price - sl) / current_price
                    
                    # Safety Check: Is SL too far? (Don't enter if risk is huge)
                    if risk_pct > MAX_SL_PERCENT:
                        print(f"⚠️ Signal Ignored: Stop Loss too wide ({risk_pct*100:.2f}%)")
                    elif sl >= current_price:
                        print(f"⚠️ Signal Ignored: SL Calculation Error (Price close to low)")
                    else:
                        msg = (
                            f"🟢 **LONG ENTRY (SMC)**\n"
                            f"Pair: `{SYMBOL}`\n"
                            f"Price: `${current_price:.2f}`\n"
                            f"---------------------\n"
                            f"🛑 **SL: `${sl:.2f}`** (Below Structure)\n"
                            f"🎯 **TP: `${tp:.2f}`** ({TARGET_RISK_REWARD}R)\n"
                            f"---------------------\n"
                            f"🤖 AI Conf: {confidence:.1f}%\n"
                            f"Risk: {risk_pct*100:.2f}%"
                        )
                        print(">>> SENDING BUY SIGNAL")
                        send_telegram(msg)
                        time.sleep(300) # Wait 5 mins before checking again

                # --- SELL LOGIC ---
                elif prediction == 2:
                    sl, tp = calculate_smart_levels("SELL", current_price, df_full, atr)
                    risk_pct = (sl - current_price) / current_price
                    
                    if risk_pct > MAX_SL_PERCENT:
                        print(f"⚠️ Signal Ignored: Stop Loss too wide ({risk_pct*100:.2f}%)")
                    elif sl <= current_price:
                        print(f"⚠️ Signal Ignored: SL Calculation Error (Price close to high)")
                    else:
                        msg = (
                            f"🔴 **SHORT ENTRY (SMC)**\n"
                            f"Pair: `{SYMBOL}`\n"
                            f"Price: `${current_price:.2f}`\n"
                            f"---------------------\n"
                            f"🛑 **SL: `${sl:.2f}`** (Above Structure)\n"
                            f"🎯 **TP: `${tp:.2f}`** ({TARGET_RISK_REWARD}R)\n"
                            f"---------------------\n"
                            f"🤖 AI Conf: {confidence:.1f}%\n"
                            f"Risk: {risk_pct*100:.2f}%"
                        )
                        print(">>> SENDING SELL SIGNAL")
                        send_telegram(msg)
                        time.sleep(300)
            
            else:
                # Idle print
                print(f"⏳ {SYMBOL}: ${current_price:.2f} | AI: {probs.argmax()} ({confidence:.1f}%) | Waiting...")

        time.sleep(10) # Fast check

    except KeyboardInterrupt:
        print("Stopping Bot...")
        break
    except Exception as e:
        print(f"Loop Error: {e}")
        time.sleep(5)