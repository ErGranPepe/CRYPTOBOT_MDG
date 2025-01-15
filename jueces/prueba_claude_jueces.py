# Añadir las importaciones necesarias
import tkinter as tk
from tkinter import ttk, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import requests
import threading
import time
import numpy as np
from datetime import datetime
from collections import deque

BINANCE_API_URL = 'https://api.binance.com/api/v3/ticker/price'
COMISION = 0.001  # 0.1% commission
INITIAL_BALANCE = 1000  # Initial USD balance
MIN_TRADE_AMOUNT = 10  # Minimum trade amount
MAX_HISTORY = 500  # Maximum historical prices to store

class Trade:
    def __init__(self, type_trade, price, amount, total_cost, timestamp):
        self.type = type_trade  # 'BUY' or 'SELL'
        self.price = price
        self.amount = amount
        self.total_cost = total_cost
        self.timestamp = timestamp
        self.profit = None  # Se establece cuando se cierra la operación

class CryptoBot:
    def __init__(self, root):
        self.balance = INITIAL_BALANCE
        self.btc_balance = 0
        self.starting_balance = INITIAL_BALANCE
        self.historical_prices = []
        self.running = False
        self.current_price = None
        self.trades_history = deque(maxlen=100)  # Mantener últimas 100 operaciones
        self.open_positions = []  # Lista de posiciones abiertas
        self.trade_stats = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_profit': 0,
            'largest_profit': 0,
            'largest_loss': 0,
            'average_profit': 0
        }
        self.last_trade_time = None
        self.min_trade_interval = 300  # 5 minutos entre operaciones
        self.position_sizing = 0.25  # Usar 25% del balance disponible por defecto
        
        self.create_ui(root)

    def create_ui(self, root):
        # [Previous UI creation code remains the same]
        # Add new frame for trade statistics
        stats_frame = ttk.LabelFrame(root, text="Trading Statistics", padding=10)
        stats_frame.pack(fill="x")

        self.stats_labels = {}
        for stat in ['Total Trades', 'Profitable Trades', 'Total Profit', 'Largest Profit', 
                    'Largest Loss', 'Average Profit']:
            ttk.Label(stats_frame, text=f"{stat}:").grid(row=len(self.stats_labels), column=0, sticky="w")
            self.stats_labels[stat] = ttk.Label(stats_frame, text="0")
            self.stats_labels[stat].grid(row=len(self.stats_labels), column=1, sticky="e")

    def calculate_position_size(self, confidence):
        """Calculate position size based on various factors"""
        base_size = self.balance * self.position_sizing
        
        # Adjust based on confidence
        confidence_factor = confidence / 100.0
        
        # Adjust based on recent performance
        if self.trades_history:
            recent_trades = list(self.trades_history)[-5:]  # Last 5 trades
            profitable_trades = sum(1 for trade in recent_trades if trade.profit and trade.profit > 0)
            performance_factor = profitable_trades / len(recent_trades)
        else:
            performance_factor = 0.5

        # Adjust based on market volatility
        if len(self.historical_prices) > 20:
            recent_prices = self.historical_prices[-20:]
            volatility = np.std(recent_prices) / np.mean(recent_prices)
            volatility_factor = max(0.2, min(1.0, 1.0 - volatility))
        else:
            volatility_factor = 0.5

        # Calculate final position size
        position_size = base_size * confidence_factor * performance_factor * volatility_factor
        
        # Ensure minimum and maximum limits
        return max(MIN_TRADE_AMOUNT, min(position_size, self.balance * 0.5))

    def should_take_profit(self, position):
        """Determine if we should take profit on a position"""
        if not self.current_price or not position.price:
            return False

        profit_percentage = ((self.current_price - position.price) / position.price) * 100
        
        # Dynamic take profit based on market conditions
        if len(self.historical_prices) > 20:
            volatility = np.std(self.historical_prices[-20:]) / np.mean(self.historical_prices[-20:])
            take_profit_threshold = max(1.5, min(5.0, volatility * 100))
        else:
            take_profit_threshold = 2.0

        return profit_percentage >= take_profit_threshold

    def should_cut_loss(self, position):
        """Determine if we should cut losses on a position"""
        if not self.current_price or not position.price:
            return False

        loss_percentage = ((position.price - self.current_price) / position.price) * 100
        
        # Dynamic stop loss based on market conditions
        if len(self.historical_prices) > 20:
            volatility = np.std(self.historical_prices[-20:]) / np.mean(self.historical_prices[-20:])
            stop_loss_threshold = max(1.0, min(3.0, volatility * 50))
        else:
            stop_loss_threshold = 2.0

        return loss_percentage >= stop_loss_threshold

    def make_decision(self, btc_price):
        if len(self.historical_prices) < 52:  # Maximum required by any indicator
            self.update_waiting_status()
            return

        # Check if enough time has passed since last trade
        if self.last_trade_time and time.time() - self.last_trade_time < self.min_trade_interval:
            return

        # Get indicator votes
        votes = []
        confidence_scores = []
        for indicator, calculate_decision in self.indicators.items():
            decision = calculate_decision(self.historical_prices)
            if decision != "WAITING" and decision != "NEUTRAL":
                votes.append(decision)
                # Add confidence score based on indicator accuracy
                confidence_scores.append(self.get_indicator_confidence(indicator))

        if not votes:
            self.final_decision_label["text"] = "NEUTRAL"
            self.final_decision_label["foreground"] = "gray"
            return

        # Calculate overall confidence
        buy_votes = votes.count("BUY")
        sell_votes = votes.count("SELL")
        total_votes = len(votes)
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0

        # Analyze open positions
        for position in self.open_positions:
            if self.should_take_profit(position):
                self.sell_btc(position.amount, "TAKE_PROFIT")
                continue
            if self.should_cut_loss(position):
                self.sell_btc(position.amount, "STOP_LOSS")
                continue

        # Make trading decision
        if total_votes > 0:
            confidence = (max(buy_votes, sell_votes) / total_votes * 100) * (avg_confidence / 100)
            
            if confidence >= 60:  # Minimum confidence threshold
                if buy_votes > sell_votes:
                    # Check if we should open a new position
                    if self.balance >= MIN_TRADE_AMOUNT and self.is_good_entry_point():
                        position_size = self.calculate_position_size(confidence)
                        self.buy_btc(position_size, btc_price)
                elif sell_votes > buy_votes:
                    # Check if we should close any positions
                    if self.btc_balance > 0 and self.is_good_exit_point():
                        self.sell_btc(self.btc_balance * 0.5, "SIGNAL")  # Sell half of holdings

    def is_good_entry_point(self):
        """Analyze if current price is a good entry point"""
        if len(self.historical_prices) < 20:
            return False

        # Check if price is near recent support levels
        recent_prices = self.historical_prices[-20:]
        min_price = min(recent_prices)
        current_price = self.historical_prices[-1]
        
        if current_price > min_price * 1.05:  # Price is more than 5% above recent low
            return False

        # Check if we're not in a strong downtrend
        ma20 = np.mean(recent_prices)
        if current_price < ma20 * 0.95:  # Price is more than 5% below MA20
            return False

        # Check volume trend if available
        # [Add volume analysis here if you have volume data]

        return True

    def is_good_exit_point(self):
        """Analyze if current price is a good exit point"""
        if len(self.historical_prices) < 20:
            return False

        # Check if price is near recent resistance levels
        recent_prices = self.historical_prices[-20:]
        max_price = max(recent_prices)
        current_price = self.historical_prices[-1]
        
        if current_price < max_price * 0.95:  # Price is more than 5% below recent high
            return False

        # Check if we're not in a strong uptrend
        ma20 = np.mean(recent_prices)
        if current_price > ma20 * 1.05:  # Price is more than 5% above MA20
            return False

        return True

    def get_indicator_confidence(self, indicator):
        """Calculate confidence score for each indicator based on historical accuracy"""
        if not self.trades_history:
            return 50  # Default confidence

        # Get recent trades that used this indicator
        relevant_trades = [t for t in self.trades_history 
                         if hasattr(t, 'indicators') and indicator in t.indicators]
        
        if not relevant_trades:
            return 50

        # Calculate success rate
        successful_trades = sum(1 for t in relevant_trades if t.profit and t.profit > 0)
        success_rate = (successful_trades / len(relevant_trades)) * 100

        return success_rate

    def update_trade_stats(self):
        """Update trading statistics"""
        if not self.trades_history:
            return

        profits = [t.profit for t in self.trades_history if t.profit is not None]
        if not profits:
            return

        self.trade_stats['total_trades'] = len(self.trades_history)
        self.trade_stats['profitable_trades'] = sum(1 for p in profits if p > 0)
        self.trade_stats['total_profit'] = sum(profits)
        self.trade_stats['largest_profit'] = max(profits)
        self.trade_stats['largest_loss'] = min(profits)
        self.trade_stats['average_profit'] = np.mean(profits)

        # Update stats labels
        for stat, label in self.stats_labels.items():
            value = self.trade_stats.get(stat.lower().replace(' ', '_'), 0)
            if 'profit' in stat.lower():
                label['text'] = f"{value:.2f}%"
            else:
                label['text'] = str(value)

    def buy_btc(self, usd_amount, btc_price):
        """Execute a buy order with improved position tracking"""
        try:
            if usd_amount < MIN_TRADE_AMOUNT:
                self.log_message("Insufficient amount for buying")
                return

            btc_amount = (usd_amount / btc_price) * (1 - COMISION)
            
            # Create new trade record
            trade = Trade('BUY', btc_price, btc_amount, usd_amount, time.time())
            
            self.btc_balance += btc_amount
            self.balance -= usd_amount
            self.open_positions.append(trade)
            self.trades_history.append(trade)
            self.last_trade_time = time.time()
            
            self.update_ui()
            self.update_trade_stats()
            self.log_message(f"BUY: {btc_amount:.8f} BTC at ${btc_price:.2f}")
            
        except Exception as e:
            self.log_message(f"Error during buy: {str(e)}")

    def sell_btc(self, btc_amount, reason="SIGNAL"):
        """Execute a sell order with profit tracking"""
        try:
            if btc_amount <= 0 or self.btc_balance < btc_amount:
                self.log_message("Insufficient BTC for selling")
                return
            
            if not self.current_price:
                self.log_message("Cannot sell: price unavailable")
                return
            
            sale_amount = btc_amount * self.current_price * (1 - COMISION)
            
            # Create new trade record
            trade = Trade('SELL', self.current_price, btc_amount, sale_amount, time.time())
            
            # Calculate profit from corresponding buy position
            if self.open_positions:
                buy_position = self.open_positions.pop(0)  # FIFO
                profit_percentage = ((sale_amount - buy_position.total_cost) / 
                                  buy_position.total_cost * 100)
                trade.profit = profit_percentage
            
            self.btc_balance -= btc_amount
            self.balance += sale_amount
            self.trades_history.append(trade)
            self.last_trade_time = time.time()
            
            self.update_ui()
            self.update_trade_stats()
            self.log_message(
                f"SELL ({reason}): {btc_amount:.8f} BTC at ${self.current_price:.2f} "
                f"Profit: {trade.profit:.2f}% Balance: ${self.balance:.2f}")
            
        except Exception as e:
            self.log_message(f"Error during sell: {str(e)}")

    def start_bot(self):
        self.running = True
        self.start_button["state"] = "disabled"
        self.pause_button["state"] = "normal"
        self.sell_button["state"] = "normal"
        self.log_message("Bot started")
        self.bot_thread = threading.Thread(target=self.run_bot)
        self.bot_thread.daemon = True  # Make thread daemon so it closes with the application
        self.bot_thread.start()

    def pause_bot(self):
        self.running = False
        self.start_button["state"] = "normal"
        self.pause_button["state"] = "disabled"
        self.log_message("Bot paused")

    def run_bot(self):
        last_error_time = None
        error_count = 0
        
        while self.running:
            try:
                btc_price = self.get_current_btc_price()
                if btc_price:
                    self.current_price = btc_price
                    self.btc_price_label["text"] = f"${btc_price:.2f}"
                    self.historical_prices.append(btc_price)

                    # Limit historical data size
                    if len(self.historical_prices) > MAX_HISTORY:
                        self.historical_prices.pop(0)

                    self.update_graph()
                    self.make_decision(btc_price)
                    self.update_ui()

                    # Reset error tracking on successful execution
                    error_count = 0
                    last_error_time = None

                # Dynamic sleep based on market volatility
                if len(self.historical_prices) >= 2:
                    price_change = abs(self.historical_prices[-1] - self.historical_prices[-2])
                    volatility_sleep = max(1, min(5, 3 - (price_change / self.historical_prices[-1]) * 100))
                    time.sleep(volatility_sleep)
                else:
                    time.sleep(2)

            except Exception as e:
                error_count += 1
                current_time = time.time()
                
                # If we've had multiple errors in a short time, increase wait time
                if last_error_time and current_time - last_error_time < 60:
                    wait_time = min(300, 5 * error_count)  # Max 5 minutes wait
                else:
                    wait_time = 5
                    error_count = 1

                self.log_message(f"Error in bot execution: {str(e)}")
                self.log_message(f"Waiting {wait_time} seconds before retry...")
                last_error_time = current_time
                time.sleep(wait_time)

    def get_current_btc_price(self):
        try:
            response = requests.get(BINANCE_API_URL, params={"symbol": "BTCUSDT"})
            response.raise_for_status()
            data = response.json()
            
            if "price" not in data:
                self.log_message("Invalid API response format")
                return None
                
            price = float(data["price"])
            
            # Validate price is reasonable (e.g., not 0 or extremely high)
            if price <= 0 or price > 1000000:  # $1M as upper limit
                self.log_message(f"Received suspicious price: ${price}")
                return None
                
            return price
        except requests.exceptions.RequestException as e:
            self.log_message(f"Network error getting price: {str(e)}")
            return None
        except ValueError as e:
            self.log_message(f"Error parsing price data: {str(e)}")
            return None
        except Exception as e:
            self.log_message(f"Unexpected error getting price: {str(e)}")
            return None

    def update_waiting_status(self):
        for label in self.decision_labels.values():
            label["text"] = "Waiting for data..."
            label["foreground"] = "gray"
        self.final_decision_label["text"] = "Waiting for sufficient data..."
        self.final_decision_label["foreground"] = "gray"

    def update_decision_label(self, label, decision):
        label["text"] = decision
        if decision == "BUY":
            label["foreground"] = "green"
        elif decision == "SELL":
            label["foreground"] = "red"
        elif decision == "NEUTRAL":
            label["foreground"] = "gray"
        else:  # WAITING
            label["foreground"] = "blue"

    def update_ui(self):
        if self.current_price:
            # Update balance displays
            self.usd_balance_label["text"] = f"${self.balance:.2f}"
            self.btc_balance_label["text"] = f"{self.btc_balance:.8f}"
            
            # Calculate total value and profit
            total_value = self.balance + (self.btc_balance * self.current_price)
            profit = ((total_value - self.starting_balance) / self.starting_balance) * 100
            
            # Update profit display with color coding
            self.profit_label["text"] = f"{profit:.2f}%"
            
            if profit >= 5:
                self.profit_label["foreground"] = "green"
            elif profit <= -5:
                self.profit_label["foreground"] = "red"
            elif profit >= 0:
                self.profit_label["foreground"] = "forest green"
            else:
                self.profit_label["foreground"] = "indian red"

    def update_graph(self):
        try:
            self.ax.clear()
            
            if len(self.historical_prices) < 2:
                return

            # Convert prices to numpy array for easier manipulation
            prices = np.array(self.historical_prices)
            x_range = np.arange(len(prices))
            
            # Plot main price line
            self.ax.plot(x_range, prices, label="BTC Price", color='black', linewidth=1.5)
            
            # Calculate and plot moving averages
            if len(prices) >= 50:
                ma50 = np.convolve(prices, np.ones(50)/50, mode='valid')
                ma50_x = np.arange(49, len(prices))  # Adjusted x range
                self.ax.plot(ma50_x, ma50, label="MA(50)", color='blue', linestyle='--', alpha=0.7)
            
            if len(prices) >= 20:
                # Calculate Bollinger Bands
                ma20 = np.convolve(prices, np.ones(20)/20, mode='valid')
                ma20_x = np.arange(19, len(prices))  # Adjusted x range
                
                # Calculate rolling standard deviation
                window_size = 20
                std20 = []
                for i in range(window_size - 1, len(prices)):
                    std20.append(np.std(prices[i-window_size+1:i+1]))
                std20 = np.array(std20)
                
                # Ensure all arrays have the same length
                upper_band = ma20 + (2 * std20)
                lower_band = ma20 - (2 * std20)
                
                # Plot Bollinger Bands
                self.ax.plot(ma20_x, upper_band, label="Upper BB", color='red', linestyle=':', alpha=0.5)
                self.ax.plot(ma20_x, lower_band, label="Lower BB", color='green', linestyle=':', alpha=0.5)
                
                # Fill between bands
                self.ax.fill_between(ma20_x, lower_band, upper_band, alpha=0.1, color='gray')

            # Customize graph appearance
            self.ax.set_title("BTC/USDT Price and Technical Indicators")
            self.ax.set_xlabel("Time (periods)")
            self.ax.set_ylabel("Price (USDT)")
            self.ax.legend(loc='upper left', framealpha=0.9)
            self.ax.grid(True, alpha=0.3)
            
            # Set dynamic y-axis limits with padding
            if len(prices) > 0:
                ymin = np.min(prices) * 0.998
                ymax = np.max(prices) * 1.002
                self.ax.set_ylim(ymin, ymax)
            
            # Format y-axis labels as currency
            self.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.2f}'))
            
            # Remove unnecessary x-axis labels to avoid crowding
            self.ax.xaxis.set_major_locator(plt.MaxNLocator(10))
            
            self.canvas.draw()
            
        except Exception as e:
            self.log_message(f"Error updating graph: {str(e)}")
            # Add more detailed error information for debugging
            import traceback
            self.log_message(f"Detailed error: {traceback.format_exc()}")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        root.title("Crypto Trading Bot - Enhanced Version")
        root.geometry("1400x800")
        
        style = ttk.Style()
        style.theme_use('default')  # Or 'clam', 'alt', 'classic' depending on your preference
        
        app = CryptoBot(root)
        root.mainloop()
    except Exception as e:
        print(f"Critical error: {str(e)}")
        raise