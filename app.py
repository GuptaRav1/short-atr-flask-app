from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from binance.um_futures import UMFutures
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import threading
import json
import os

app = Flask(__name__)

class ATRScanner:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize the ATR Scanner for Binance UM Futures
        
        Args:
            api_key: Binance API key (optional for market data)
            api_secret: Binance API secret (optional for market data)
        """
        # Initialize client (no credentials needed for market data)
        self.client = UMFutures(key=api_key, secret=api_secret)
        
    def get_active_symbols(self) -> List[str]:
        """Get all active USDT perpetual futures symbols"""
        try:
            exchange_info = self.client.exchange_info()
            symbols = []
            
            for symbol_info in exchange_info['symbols']:
                # Filter for USDT perpetual futures that are trading
                if (symbol_info['symbol'].endswith('USDT') and 
                    symbol_info['contractType'] == 'PERPETUAL' and
                    symbol_info['status'] == 'TRADING'):
                    symbols.append(symbol_info['symbol'])
                    
            return symbols
        except Exception as e:
            print(f"Error getting symbols: {e}")
            return []
    
    def get_kline_data(self, symbol: str, interval: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Get kline/candlestick data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Kline interval (default: '1h')
            limit: Number of data points to retrieve (default: 100)
        """
        try:
            klines = self.client.klines(symbol=symbol, interval=interval, limit=limit)
            
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to appropriate data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            return df
            
        except Exception as e:
            print(f"Error getting kline data for {symbol}: {e}")
            return None
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 21) -> float:
        """
        Calculate Average True Range (ATR)
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period (default: 21)
        """
        if len(df) < period + 1:
            return 0.0
            
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR as simple moving average of True Range
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else 0.0
    
    def calculate_atr_percentage(self, atr_value: float, current_price: float, multiplier: float = 1.5) -> float:
        """
        Calculate ATR as percentage of current price
        
        Args:
            atr_value: ATR value
            current_price: Current price of the asset
            multiplier: ATR multiplier (default: 1.5)
        """
        if current_price <= 0:
            return 0.0
            
        atr_adjusted = atr_value * multiplier
        atr_percentage = (atr_adjusted / current_price) * 100
        
        return atr_percentage
    
    def scan_symbols(self, min_atr_percentage: float = 0.1, atr_period: int = 50, 
                    atr_multiplier: float = 0.5, interval: str = '1m') -> List[Dict]:
        """
        Scan symbols for ATR percentage criteria
        
        Args:
            min_atr_percentage: Minimum ATR percentage threshold (default: 0.1%)
            atr_period: ATR calculation period (default: 21)
            atr_multiplier: ATR multiplier (default: 1.5)
            interval: Kline interval for analysis (default: '1h')
        """
        print("Getting active symbols...")
        symbols = self.get_active_symbols()
        print(f"Found {len(symbols)} active symbols")
        
        results = []
        
        for i, symbol in enumerate(symbols):
            try:
                print(f"Processing {symbol} ({i+1}/{len(symbols)})...")
                
                # Get kline data
                df = self.get_kline_data(symbol, interval=interval, limit=atr_period + 50)
                if df is None or len(df) < atr_period + 1:
                    continue
                
                # Calculate ATR
                atr_value = self.calculate_atr(df, period=atr_period)
                current_price = float(df['close'].iloc[-1])
                
                # Calculate ATR percentage
                atr_percentage = self.calculate_atr_percentage(atr_value, current_price, atr_multiplier)
                
                # Check if meets criteria
                if atr_percentage >= min_atr_percentage:
                    result = {
                        'symbol': symbol,
                        'current_price': current_price,
                        'atr_value': atr_value,
                        'atr_percentage': atr_percentage,
                        'atr_period': atr_period,
                        'atr_multiplier': atr_multiplier
                    }
                    results.append(result)
                    print(f" {symbol}: {atr_percentage:.3f}% ATR")
                
                # Rate limiting to avoid API limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        return results

# Global variables for storing scan results and status
scan_data = {
    'results': [],
    'last_update': None,
    'scanning': False,
    'scan_settings': {
        'min_atr_percentage': 0.5,
        'atr_period': 50,
        'atr_multiplier': 0.5,
        'interval': '1m'
    }
}

scanner = ATRScanner()

def perform_scan():
    """Perform ATR scan and update global data"""
    global scan_data
    
    try:
        scan_data['scanning'] = True
        print(f"Starting ATR scan at {datetime.now()}")
        
        results = scanner.scan_symbols(
            min_atr_percentage=scan_data['scan_settings']['min_atr_percentage'],
            atr_period=scan_data['scan_settings']['atr_period'],
            atr_multiplier=scan_data['scan_settings']['atr_multiplier'],
            interval=scan_data['scan_settings']['interval']
        )
        
        # Sort results by ATR percentage descending
        results.sort(key=lambda x: x['atr_percentage'], reverse=True)
        
        scan_data['results'] = results
        scan_data['last_update'] = datetime.now()
        scan_data['scanning'] = False
        
        print(f"Scan completed. Found {len(results)} symbols.")
        
    except Exception as e:
        print(f"Error during scan: {e}")
        scan_data['scanning'] = False

def schedule_scans():
    """Background thread to schedule scans every hour"""
    while True:
        perform_scan()
        # Wait for 1 hour (3600 seconds)
        time.sleep(3600)

# Start background scanning thread
scan_thread = threading.Thread(target=schedule_scans, daemon=True)
scan_thread.start()

@app.route('/')
def index():
    """Main page displaying ATR scan results"""
    return render_template('index.html')

@app.route('/api/scan-data')
def get_scan_data():
    """API endpoint to get current scan data"""
    response_data = {
        'results': scan_data['results'],
        'last_update': scan_data['last_update'].isoformat() if scan_data['last_update'] else None,
        'scanning': scan_data['scanning'],
        'total_symbols': len(scan_data['results']),
        'scan_settings': scan_data['scan_settings']
    }
    return jsonify(response_data)

@app.route('/api/force-scan', methods=['POST'])
def force_scan():
    """API endpoint to force a new scan"""
    if not scan_data['scanning']:
        # Start scan in background thread
        threading.Thread(target=perform_scan, daemon=True).start()
        return jsonify({'status': 'scan_started'})
    else:
        return jsonify({'status': 'scan_already_running'})

@app.route('/api/update-settings', methods=['POST'])
def update_settings():
    """API endpoint to update scan settings"""
    try:
        new_settings = request.json
        
        # Validate settings
        if 'min_atr_percentage' in new_settings:
            scan_data['scan_settings']['min_atr_percentage'] = float(new_settings['min_atr_percentage'])
        if 'atr_period' in new_settings:
            scan_data['scan_settings']['atr_period'] = int(new_settings['atr_period'])
        if 'atr_multiplier' in new_settings:
            scan_data['scan_settings']['atr_multiplier'] = float(new_settings['atr_multiplier'])
        if 'interval' in new_settings:
            scan_data['scan_settings']['interval'] = new_settings['interval']
        
        return jsonify({'status': 'settings_updated', 'settings': scan_data['scan_settings']})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/export-symbols')
def export_symbols():
    """API endpoint to export symbols in Binance format"""
    if not scan_data['results']:
        return jsonify({'symbols': '', 'count': 0})
    
    # Sort by ATR percentage descending
    results = sorted(scan_data['results'], key=lambda x: x['atr_percentage'], reverse=True)
    
    # Create list of symbols in BINANCE:SYMBOL.P format
    symbol_list = []
    for result in results:
        symbol = result['symbol']
        binance_format = f"BINANCE:{symbol}.P"
        symbol_list.append(binance_format)
    
    # Join all symbols with commas
    symbols_string = ','.join(symbol_list)
    
    return jsonify({'symbols': symbols_string, 'count': len(symbol_list)})

# Create templates directory if it doesn't exist
if not os.path.exists('templates'):
    os.makedirs('templates')

if __name__ == '__main__':
    print("Starting ATR Scanner Flask App...")
    print("The app will automatically scan every hour.")
    print("Access the web interface at: http://localhost:5000")
    
    # Perform initial scan
    threading.Thread(target=perform_scan, daemon=True).start()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
