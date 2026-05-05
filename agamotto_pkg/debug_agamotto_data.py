
import sys
import os
import asyncio
import logging
import pandas as pd
from datetime import datetime

# Setup paths - prioritize gauntlet package
gauntlet_path = os.path.join(os.getcwd(), 'gauntlet', 'agamotto_pkg', 'src')
if os.path.exists(gauntlet_path):
    sys.path.insert(0, gauntlet_path)
    print(f"Added {gauntlet_path} to sys.path")

# Mock config
config = {
    "SYMBOLS": ["BTCUSDT"],
    "TIMEFRAME": "1h",
    "kline_limit": 100
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_agamotto_data():
    try:
        from agamotto.trading import Agamotto
        
        print(f"Initializing Agamotto at {datetime.now()}...")
        # Initialize Agamotto (assuming standard init signature)
        # We might need to mock some parts if it requires full environment
        
        # Try to find a class that looks like the main one if import fails or differs
        # But commonly it's Agamotto class in trading module
        
        agamotto = Agamotto(
            symbols=config["SYMBOLS"],
            timeframe=config["TIMEFRAME"]
        )
        
        print("Loading data...")
        await agamotto.load_data(limit=10)
        
        print("\n--- Last 5 Data Rows for BTCUSDT ---")
        if "BTCUSDT" in agamotto.data:
            df = agamotto.data["BTCUSDT"]
            print(df.tail())
            
            # Check last timestamp
            last_time = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else df.iloc[-1]['timestamp'] if 'timestamp' in df else "Unknown"
            print(f"\nLast Timestamp: {last_time}")
            
            # Check if it's the current hour
            now = datetime.now()
            print(f"Current Time: {now}")
            
        else:
            print("No data found for BTCUSDT")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agamotto_data())
