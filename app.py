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

# HTML template content
html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ATR Scanner - Binance Futures</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5rem;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .status-bar {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }
        .status-info {
            display: flex;
            gap: 30px;
            align-items: center;
        }
        .status-item {
            display: flex;
            flex-direction: column;
        }
        .status-label {
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 5px;
        }
        .status-value {
            font-weight: bold;
            font-size: 1.1rem;
        }
        .scanning {
            color: #ff6b35;
        }
        .updated {
            color: #28a745;
        }
        .controls {
            display: flex;
            gap: 10px;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover {
            background: #5a67d8;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .settings-panel {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .settings-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }
        .setting-item {
            display: flex;
            flex-direction: column;
        }
        .setting-item label {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .setting-item input, .setting-item select {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .results-container {
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .results-header {
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #dee2e6;
        }
        .results-header h2 {
            margin: 0;
            color: #333;
        }
        .table-container {
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
            color: #495057;
            position: sticky;
            top: 0;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .symbol-cell {
            font-weight: bold;
            color: #667eea;
        }
        .price-cell {
            font-family: monospace;
        }
        .atr-percentage {
            font-weight: bold;
            color: #28a745;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        .no-results {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        .export-section {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .export-section h3 {
            margin: 0 0 10px 0;
            color: #1976d2;
        }
        .export-text {
            width: 100%;
            height: 100px;
            margin: 10px 0;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 ATR Scanner</h1>
            <p>Binance Futures - Real-time ATR Analysis</p>
        </div>

        <div class="status-bar">
            <div class="status-info">
                <div class="status-item">
                    <div class="status-label">Status</div>
                    <div class="status-value" id="scan-status">Loading...</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Last Update</div>
                    <div class="status-value" id="last-update">Never</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Total Symbols</div>
                    <div class="status-value" id="total-symbols">0</div>
                </div>
            </div>
            <div class="controls">
                <button onclick="forceScan()" id="force-scan-btn">Force Scan</button>
                <button onclick="toggleSettings()" id="settings-btn">Settings</button>
                <button onclick="exportSymbols()">Export Symbols</button>
            </div>
        </div>

        <div class="settings-panel" id="settings-panel" style="display: none;">
            <h3>Scan Settings</h3>
            <div class="settings-grid">
                <div class="setting-item">
                    <label>Min ATR Percentage (%)</label>
                    <input type="number" id="min-atr" step="0.1" min="0" value="0.5">
                </div>
                <div class="setting-item">
                    <label>ATR Period</label>
                    <input type="number" id="atr-period" min="1" max="200" value="50">
                </div>
                <div class="setting-item">
                    <label>ATR Multiplier</label>
                    <input type="number" id="atr-multiplier" step="0.1" min="0" value="0.5">
                </div>
                <div class="setting-item">
                    <label>Interval</label>
                    <select id="interval">
                        <option value="1m">1 minute</option>
                        <option value="3m">3 minutes</option>
                        <option value="5m">5 minutes</option>
                        <option value="15m">15 minutes</option>
                        <option value="30m">30 minutes</option>
                        <option value="1h">1 hour</option>
                        <option value="2h">2 hours</option>
                        <option value="4h">4 hours</option>
                        <option value="6h">6 hours</option>
                        <option value="8h">8 hours</option>
                        <option value="12h">12 hours</option>
                        <option value="1d">1 day</option>
                    </select>
                </div>
            </div>
            <button onclick="updateSettings()">Update Settings</button>
        </div>

        <div class="export-section" id="export-section" style="display: none;">
            <h3>Export Symbols (TradingView Format)</h3>
            <p>Copy the text below and paste into TradingView watchlist:</p>
            <textarea class="export-text" id="export-text" readonly></textarea>
        </div>

        <div class="results-container">
            <div class="results-header">
                <h2>ATR Scan Results</h2>
            </div>
            <div id="results-content">
                <div class="loading">Loading scan results...</div>
            </div>
        </div>
    </div>

    <script>
        let settingsVisible = false;
        let exportVisible = false;

        function loadScanData() {
            fetch('/api/scan-data')
                .then(response => response.json())
                .then(data => {
                    updateStatus(data);
                    updateResults(data.results);
                })
                .catch(error => {
                    console.error('Error loading scan data:', error);
                    document.getElementById('results-content').innerHTML = 
                        '<div class="no-results">Error loading data. Please try again.</div>';
                });
        }

        function updateStatus(data) {
            const statusEl = document.getElementById('scan-status');
            const lastUpdateEl = document.getElementById('last-update');
            const totalSymbolsEl = document.getElementById('total-symbols');

            if (data.scanning) {
                statusEl.textContent = 'Scanning...';
                statusEl.className = 'status-value scanning';
            } else {
                statusEl.textContent = 'Ready';
                statusEl.className = 'status-value updated';
            }

            if (data.last_update) {
                const updateTime = new Date(data.last_update);
                lastUpdateEl.textContent = updateTime.toLocaleString();
            }

            totalSymbolsEl.textContent = data.total_symbols;

            // Update settings form
            if (data.scan_settings) {
                document.getElementById('min-atr').value = data.scan_settings.min_atr_percentage;
                document.getElementById('atr-period').value = data.scan_settings.atr_period;
                document.getElementById('atr-multiplier').value = data.scan_settings.atr_multiplier;
                document.getElementById('interval').value = data.scan_settings.interval;
            }

            // Enable/disable force scan button
            const forceScanBtn = document.getElementById('force-scan-btn');
            forceScanBtn.disabled = data.scanning;
            forceScanBtn.textContent = data.scanning ? 'Scanning...' : 'Force Scan';
        }

        function updateResults(results) {
            const contentEl = document.getElementById('results-content');

            if (!results || results.length === 0) {
                contentEl.innerHTML = '<div class="no-results">No symbols found meeting the criteria.</div>';
                return;
            }

            let tableHTML = `
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Current Price</th>
                                <th>ATR Value</th>
                                <th>ATR %</th>
                                <th>Period</th>
                                <th>Multiplier</th>
                            </tr>
                        </thead>
                        <tbody>
            `;

            results.forEach(result => {
                tableHTML += `
                    <tr>
                        <td class="symbol-cell">${result.symbol}</td>
                        <td class="price-cell">$${result.current_price.toFixed(4)}</td>
                        <td>${result.atr_value.toFixed(6)}</td>
                        <td class="atr-percentage">${result.atr_percentage.toFixed(3)}%</td>
                        <td>${result.atr_period}</td>
                        <td>${result.atr_multiplier}</td>
                    </tr>
                `;
            });

            tableHTML += '</tbody></table></div>';
            contentEl.innerHTML = tableHTML;
        }

        function forceScan() {
            fetch('/api/force-scan', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'scan_started') {
                        loadScanData();
                    }
                })
                .catch(error => console.error('Error forcing scan:', error));
        }

        function toggleSettings() {
            settingsVisible = !settingsVisible;
            const panel = document.getElementById('settings-panel');
            const btn = document.getElementById('settings-btn');
            
            if (settingsVisible) {
                panel.style.display = 'block';
                btn.textContent = 'Hide Settings';
            } else {
                panel.style.display = 'none';
                btn.textContent = 'Settings';
            }
        }

        function updateSettings() {
            const settings = {
                min_atr_percentage: parseFloat(document.getElementById('min-atr').value),
                atr_period: parseInt(document.getElementById('atr-period').value),
                atr_multiplier: parseFloat(document.getElementById('atr-multiplier').value),
                interval: document.getElementById('interval').value
            };

            fetch('/api/update-settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(settings)
            })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'settings_updated') {
                        alert('Settings updated successfully!');
                    }
                })
                .catch(error => console.error('Error updating settings:', error));
        }

        function exportSymbols() {
            fetch('/api/export-symbols')
                .then(response => response.json())
                .then(data => {
                    const exportSection = document.getElementById('export-section');
                    const exportText = document.getElementById('export-text');
                    
                    exportText.value = data.symbols;
                    exportVisible = !exportVisible;
                    
                    if (exportVisible) {
                        exportSection.style.display = 'block';
                    } else {
                        exportSection.style.display = 'none';
                    }
                })
                .catch(error => console.error('Error exporting symbols:', error));
        }

        // Load data on page load
        loadScanData();

        // Auto-refresh every 30 seconds
        setInterval(loadScanData, 30000);
    </script>
</body>
</html>'''

# Write the HTML template to file
with open('templates/index.html', 'w') as f:
    f.write(html_template)

if __name__ == '__main__':
    print("Starting ATR Scanner Flask App...")
    print("The app will automatically scan every hour.")
    print("Access the web interface at: http://localhost:5000")
    
    # Perform initial scan
    threading.Thread(target=perform_scan, daemon=True).start()
    
    app.run(debug=True, host='0.0.0.0', port=5000)