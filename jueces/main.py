import tkinter as tk
from tkinter import ttk, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import requests
import threading
import time
import numpy as np
from datetime import datetime

BINANCE_API_URL = 'https://api.binance.com/api/v3/ticker/price'
COMISION = 0.001  # 0.1% commission
INITIAL_BALANCE = 1000  # Initial USD balance
MIN_TRADE_AMOUNT = 10  # Minimum trade amount
MAX_HISTORY = 500  # Maximum historical prices to store
class SupremeJudge:
    def __init__(self):
        self.voting_threshold = 0.5  # Umbral de votos para decidir

    def make_final_decision(self, votes):
        buy_votes = votes.count("BUY")
        sell_votes = votes.count("SELL")
        total_votes = len(votes)

        if total_votes == 0:
            return "WAITING"

        buy_ratio = buy_votes / total_votes
        sell_ratio = sell_votes / total_votes

        if buy_ratio > self.voting_threshold:
            return "BUY"
        elif sell_ratio > self.voting_threshold:
            return "SELL"
        else:
            return "WAITING"

class CryptoBot:
    def __init__(self, root):
        self.balance = INITIAL_BALANCE
        self.btc_balance = 0
        self.starting_balance = INITIAL_BALANCE
        self.historical_prices = []
        self.running = False
        self.current_price = None
        self.last_trade_price = None
        self.consecutive_losses = 0
        self.trade_cooldown = 0
        self.stop_loss_percentage = 0.95  # 5% de pérdida máxima
        self.take_profit_percentage = 1.05  # 5% de ganancia mínima
        self.open_positions = []  # Lista de posiciones abiertas
        self.trade_cooldown_time = 50  # 5 minutos de cooldown
        self.last_trade_time = None  # Tiempo de la última operación
        self.market_volatility_threshold = 0.02  # 2% de volatilidad
        self.create_ui(root)

    def log_performance(self):
        """
        Actualiza las métricas de rendimiento en la interfaz gráfica.
        """
        total_profit = 0
        total_trades = 0

        # Calcular el beneficio total y el número de operaciones cerradas
        for position in self.open_positions:
            if position["status"] == "closed":
                profit = (position["sell_price"] - position["buy_price"]) * position["btc_amount"]
                total_profit += profit
                total_trades += 1

        # Actualizar las etiquetas en la interfaz gráfica
        self.total_profit_label["text"] = f"${total_profit:.2f}"
        self.total_trades_label["text"] = f"{total_trades}"
        self.current_balance_label["text"] = f"${self.balance:.2f}"
        self.btc_balance_label["text"] = f"{self.btc_balance:.8f} BTC"

    def find_local_min_max(self, prices, window=20):
        """
        Encuentra los mínimos y máximos locales en una serie de precios.
        :param prices: Lista de precios históricos.
        :param window: Tamaño de la ventana para buscar mínimos y máximos.
        :return: Lista de tuplas (índice, precio, tipo) donde tipo es "min" o "max".
        """
        local_min_max = []
        for i in range(window, len(prices) - window):
            local_window = prices[i - window:i + window + 1]
            if prices[i] == min(local_window):
                local_min_max.append((i, prices[i], "min"))
            elif prices[i] == max(local_window):
                local_min_max.append((i, prices[i], "max"))
        return local_min_max
    
    def should_buy(self, btc_price):
        """
        Decide si es un buen momento para comprar basado en mínimos locales.
        """
        if len(self.historical_prices) < 50:  # Necesitamos suficientes datos
            return False

        # Encontrar mínimos locales
        local_min_max = self.find_local_min_max(self.historical_prices)
        last_min = None
        for idx, price, type_ in local_min_max:
            if type_ == "min":
                last_min = price

        # Comprar si el precio actual está cerca del último mínimo local
        if last_min and btc_price <= last_min * 1.01:  # 1% de margen
            return True
        return False

    def should_sell(self, btc_price):
        """
        Decide si es un buen momento para vender basado en máximos locales.
        """
        if len(self.historical_prices) < 50:  # Necesitamos suficientes datos
            return False

        # Encontrar máximos locales
        local_min_max = self.find_local_min_max(self.historical_prices)
        last_max = None
        for idx, price, type_ in local_min_max:
            if type_ == "max":
                last_max = price

        # Vender si el precio actual está cerca del último máximo local
        if last_max and btc_price >= last_max * 0.99:  # 1% de margen
            return True
        return False

    def monitor_market(self):
        """
        Monitorea la volatilidad del mercado y ajusta la estrategia de trading.
        """
        current_volatility = self.calculate_market_volatility()
        if current_volatility > self.market_volatility_threshold:
            self.adjust_trading_strategy(current_volatility)
            self.log_message(f"Market volatility high: {current_volatility:.2%}. Adjusting strategy.")

    def adjust_trading_strategy(self, volatility):
        """
        Ajusta la estrategia de trading según la volatilidad del mercado.
        """
        if volatility > self.market_volatility_threshold:
            # Reducir el tamaño de las operaciones en mercados volátiles
            self.min_trade_amount = 50  # Reducir el monto mínimo de operación
            self.max_trade_amount = 100  # Reducir el monto máximo de operación
            self.stop_loss_percentage = 0.03  # Aumentar el stop-loss para evitar pérdidas
            self.take_profit_percentage = 0.06  # Aumentar el take-profit para aprovechar movimientos grandes
            self.log_message("Trading strategy adjusted for high volatility.")
        else:
            # Restaurar valores predeterminados en mercados estables
            self.min_trade_amount = 100
            self.max_trade_amount = 200
            self.stop_loss_percentage = 0.02
            self.take_profit_percentage = 0.05
            self.log_message("Trading strategy adjusted for low volatility.")

    def calculate_market_volatility(self):
        """
        Calcula la volatilidad del mercado basada en los precios históricos.
        La volatilidad se define como la desviación estándar de los precios.
        """
        if len(self.historical_prices) < 2:
            return 0  # No hay suficientes datos para calcular la volatilidad

        # Convertir la lista de precios a un array de numpy para cálculos más eficientes
        prices = np.array(self.historical_prices)

        # Calcular la desviación estándar de los precios
        volatility = np.std(prices)

        # Normalizar la volatilidad respecto al precio actual
        if len(prices) > 0:
            current_price = prices[-1]
            normalized_volatility = volatility / current_price
            return normalized_volatility
        else:
            return 0
    def should_trade(self, btc_price):
        """
        Decide si se debe realizar una operación basada en la volatilidad del mercado.
        """
        volatility = self.calculate_market_volatility()
        if volatility > self.volatility_threshold:  # Umbral de volatilidad
            return False  # No operar en alta volatilidad
        return True
        
    def log_performance(self):
        """
        Registra el rendimiento del bot después de cada operación.
        """
        total_profit = 0
        total_trades = 0

        # Calcular el beneficio total y el número de operaciones cerradas
        for position in self.open_positions:
            if position["status"] == "closed":
                profit = (position["sell_price"] - position["buy_price"]) * position["btc_amount"]
                total_profit += profit
                total_trades += 1

        # Mostrar el rendimiento en el log
        self.log_message(f"Total profit: ${total_profit:.2f}")
        self.log_message(f"Total trades: {total_trades}")
        self.log_message(f"Current balance: ${self.balance:.2f}")
    def create_ui(self, root):
        root.title("Crypto Auto-Trading Bot")
        
        # Crear main container
        main_container = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for trading interface
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=2)
        
        # Right panel for performance metrics
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=1)
        
        self.create_left_panel(left_panel)
        self.create_right_panel(right_panel)

    def create_right_panel(self, parent):
        """
        Crea el panel derecho para mostrar métricas de rendimiento.
        """
        performance_frame = ttk.LabelFrame(parent, text="Performance Metrics", padding=10)
        performance_frame.pack(fill="both", expand=True)

        # Etiquetas para mostrar el rendimiento
        ttk.Label(performance_frame, text="Total Profit:").grid(row=0, column=0, sticky="w")
        self.total_profit_label = ttk.Label(performance_frame, text="$0.00")
        self.total_profit_label.grid(row=0, column=1, sticky="e")

        ttk.Label(performance_frame, text="Total Trades:").grid(row=1, column=0, sticky="w")
        self.total_trades_label = ttk.Label(performance_frame, text="0")
        self.total_trades_label.grid(row=1, column=1, sticky="e")

        ttk.Label(performance_frame, text="Current Balance:").grid(row=2, column=0, sticky="w")
        self.current_balance_label = ttk.Label(performance_frame, text="$0.00")
        self.current_balance_label.grid(row=2, column=1, sticky="e")

        ttk.Label(performance_frame, text="BTC Balance:").grid(row=3, column=0, sticky="w")
        self.btc_balance_label = ttk.Label(performance_frame, text="0.00000000 BTC")
        self.btc_balance_label.grid(row=3, column=1, sticky="e")
        
    def create_left_panel(self, parent):
        # Account frame
        account_frame = ttk.LabelFrame(parent, text="Account", padding=10)
        account_frame.pack(fill="x")

        ttk.Label(account_frame, text="USD Balance:").grid(row=0, column=0, sticky="w")
        self.usd_balance_label = ttk.Label(account_frame, text=f"{self.balance:.2f}")
        self.usd_balance_label.grid(row=0, column=1, sticky="e")

        ttk.Label(account_frame, text="BTC Balance:").grid(row=1, column=0, sticky="w")
        self.btc_balance_label = ttk.Label(account_frame, text=f"{self.btc_balance:.8f}")
        self.btc_balance_label.grid(row=1, column=1, sticky="e")

        ttk.Label(account_frame, text="Profit:").grid(row=2, column=0, sticky="w")
        self.profit_label = ttk.Label(account_frame, text="0.00%")
        self.profit_label.grid(row=2, column=1, sticky="e")

        # Actions frame
        actions_frame = ttk.Frame(parent, padding=10)
        actions_frame.pack(fill="x")

        self.start_button = ttk.Button(actions_frame, text="Start Bot", command=self.start_bot)
        self.start_button.grid(row=0, column=0, padx=5)

        self.pause_button = ttk.Button(actions_frame, text="Pause Bot", command=self.pause_bot, state="disabled")
        self.pause_button.grid(row=0, column=1, padx=5)

        self.sell_button = ttk.Button(actions_frame, text="Sell All", command=self.sell_all, state="disabled")
        self.sell_button.grid(row=0, column=2, padx=5)

        # Live data frame
        live_data_frame = ttk.LabelFrame(parent, text="Live Data", padding=10)
        live_data_frame.pack(fill="x")

        ttk.Label(live_data_frame, text="BTC/USDT Price:").grid(row=0, column=0, sticky="w")
        self.btc_price_label = ttk.Label(live_data_frame, text="Loading...")
        self.btc_price_label.grid(row=0, column=1, sticky="e")

        # Graph frame
        graph_frame = ttk.LabelFrame(parent, text="Indicators and Decisions", padding=10)
        graph_frame.pack(fill="both", expand=True)

        self.figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Decision frame
        self.create_decision_frame(parent)

    def create_right_panel(self, parent):
        # Log frame
        log_frame = ttk.LabelFrame(parent, text="Trading Log", padding=10)
        log_frame.pack(fill="both", expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(fill="both", expand=True)

        # Open positions frame
        open_positions_frame = ttk.LabelFrame(parent, text="Open Positions", padding=10)
        open_positions_frame.pack(fill="both", expand=True)

        self.open_positions_list = tk.Listbox(open_positions_frame, height=5)
        self.open_positions_list.pack(fill="both", expand=True)

        # Closed positions frame
        closed_positions_frame = ttk.LabelFrame(parent, text="Closed Positions", padding=10)
        closed_positions_frame.pack(fill="both", expand=True)

        self.closed_positions_list = tk.Listbox(closed_positions_frame, height=5)
        self.closed_positions_list.pack(fill="both", expand=True)

    def update_closed_positions_list(self):
        self.closed_positions_list.delete(0, tk.END)  # Limpiar la lista
        for position in self.open_positions:
            if position["status"] == "closed":
                buy_price = position["buy_price"]
                sell_price = position["sell_price"]
                btc_amount = position["btc_amount"]
                buy_time = position["buy_time"].strftime("%Y-%m-%d %H:%M:%S")
                sell_time = position["sell_time"].strftime("%Y-%m-%d %H:%M:%S")
                profit = ((sell_price - buy_price) / buy_price) * 100
                self.closed_positions_list.insert(tk.END, f"{buy_time} -> {sell_time} | {btc_amount:.8f} BTC | Profit: {profit:.2f}%")
        
    def update_open_positions_list(self):
    
        self.open_positions_list.delete(0, tk.END)  # Limpiar la lista
        for position in self.open_positions:
            if position["status"] == "open":
                buy_price = position["buy_price"]
                btc_amount = position["btc_amount"]
                buy_time = position["buy_time"].strftime("%Y-%m-%d %H:%M:%S")
                self.open_positions_list.insert(tk.END, f"{buy_time} | {btc_amount:.8f} BTC @ ${buy_price:.2f}")

    def create_decision_frame(self, parent):
        decision_frame = ttk.LabelFrame(parent, text="Trading Judges", padding=10)
        decision_frame.pack(fill="x")

        self.decision_labels = {}
        self.indicators = {
            "MA": self.calculate_ma_decision,
            "RSI": self.calculate_rsi_decision,
            "Bollinger": self.calculate_bollinger_decision,
            "MACD": self.calculate_macd_decision,
            "Stochastic": self.calculate_stochastic_decision,
            "Fibonacci": self.calculate_fibonacci_decision,
            "Ichimoku": self.calculate_ichimoku_decision,
            "Volume": self.calculate_volume_decision,
            "Aroon": self.calculate_aroon_decision
        }

        for i, (indicator, _) in enumerate(self.indicators.items()):
            ttk.Label(decision_frame, text=f"{indicator} Decision:").grid(row=i, column=0, sticky="w")
            self.decision_labels[indicator] = ttk.Label(decision_frame, text="Waiting...", foreground="gray")
            self.decision_labels[indicator].grid(row=i, column=1, sticky="e")

        ttk.Label(decision_frame, text="Supreme Judge Decision:").grid(row=len(self.indicators), column=0, sticky="w")
        self.final_decision_label = ttk.Label(decision_frame, text="Waiting...", foreground="gray")
        self.final_decision_label.grid(row=len(self.indicators), column=1, sticky="e")

    def log_message(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    # Trading Indicators Calculations
    def calculate_ma_decision(self, prices):
        """
        Decisión basada en la media móvil (MA).
        """
        if len(prices) < 50:
            return "WAITING"
        ma = np.mean(prices[-50:])
        return "BUY" if prices[-1] < ma else "SELL"

    def calculate_rsi_decision(self, prices, period=14):
        """
        Decisión basada en el Índice de Fuerza Relativa (RSI).
        """
        if len(prices) < period:
            return "WAITING"
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            return "NEUTRAL"
                
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return "BUY" if rsi < 30 else "SELL" if rsi > 70 else "NEUTRAL"

    def calculate_bollinger_decision(self, prices):
        if len(prices) < 20:
            return "WAITING"
                
        sma = np.mean(prices[-20:])
        std_dev = np.std(prices[-20:])
        upper_band = sma + (2 * std_dev)
        lower_band = sma - (2 * std_dev)
        current_price = prices[-1]
        
        return "BUY" if current_price < lower_band else "SELL" if current_price > upper_band else "NEUTRAL"

    def calculate_macd_decision(self, prices):
        if len(prices) < 26:
            return "WAITING"
                
        exp1 = np.exp(np.linspace(-1, 0, 12))
        exp2 = np.exp(np.linspace(-1, 0, 26))
        exp1 = exp1 / exp1.sum()
        exp2 = exp2 / exp2.sum()
        
        ema12 = np.convolve(prices, exp1, mode='full')[:len(prices)]
        ema26 = np.convolve(prices, exp2, mode='full')[:len(prices)]
        
        macd = ema12 - ema26
        signal_line = np.mean(macd[-9:])  # Signal line is the EMA of the MACD
        
        return "BUY" if macd[-1] > signal_line else "SELL" if macd[-1] < signal_line else "NEUTRAL"

    def calculate_stochastic_decision(self, prices):
        if len(prices) < 14:
            return "WAITING"
            
        low_14 = min(prices[-14:])
        high_14 = max(prices[-14:])
        
        if high_14 == low_14:
            return "NEUTRAL"
            
        k = 100 * (prices[-1] - low_14) / (high_14 - low_14)
        
        return "BUY" if k < 20 else "SELL" if k > 80 else "NEUTRAL"

    def calculate_fibonacci_decision(self, prices):
        if len(prices) < 100:
            return "WAITING"
            
        high = max(prices[-100:])
        low = min(prices[-100:])
        diff = high - low
        
        fib_38 = high - (0.382 * diff)
        fib_62 = high - (0.618 * diff)
        current_price = prices[-1]
        
        if current_price < fib_62:
            return "BUY"
        elif current_price > fib_38:
            return "SELL"
        return "NEUTRAL"

    def calculate_ichimoku_decision(self, prices):
        if len(prices) < 52:
            return "WAITING"
            
        tenkan_sen = (max(prices[-9:]) + min(prices[-9:])) / 2
        kijun_sen = (max(prices[-26:]) + min(prices[-26:])) / 2
        
        return "BUY" if tenkan_sen > kijun_sen else "SELL"

    def calculate_volume_decision(self, prices):
        if len(prices) < 20:
            return "WAITING"
            
        price_change = prices[-1] - prices[-2]
        volume_ma = np.mean(np.abs(np.diff(prices[-20:])))
        
        if abs(price_change) > volume_ma * 1.5:
            return "BUY" if price_change > 0 else "SELL"
        return "NEUTRAL"

    def calculate_aroon_decision(self, prices):
        if len(prices) < 25:
            return "WAITING"
            
        period = 25
        high_idx = prices[-period:].index(max(prices[-period:]))
        low_idx = prices[-period:].index(min(prices[-period:]))
        
        aroon_up = ((period - high_idx) / period) * 100
        aroon_down = ((period - low_idx) / period) * 100
        
        return "BUY" if aroon_up > aroon_down else "SELL"

    def make_decision(self, btc_price):
        if len(self.historical_prices) < 52:  # Maximum required by any indicator
            self.update_waiting_status()
            return
        # Calcular la volatilidad del mercado
        current_volatility = self.calculate_market_volatility()

        # Evitar operar en merc ados altamente volátiles
        if current_volatility > self.market_volatility_threshold:
            self.log_message("High market volatility detected. Avoiding new trades.")
            return  # No realizar nuevas operaciones
        
       # Verificar si es un buen momento para comprar
        if self.should_buy(btc_price):
            self.buy_btc(btc_price)
            return  # Salir del ciclo después de comprar

        # Verificar si es un buen momento para vender
        if self.should_sell(btc_price):
            self.sell_all()
            return  # Salir del ciclo después de vender

        # Actualizar la lista de precios históricos
        self.historical_prices.append(btc_price)
        if len(self.historical_prices) > MAX_HISTORY:
            self.historical_prices.pop(0)

        if self.last_trade_time and (time.time() - self.last_trade_time) < self.trade_cooldown_time:
            self.log_message(f"Cooldown active. Waiting to trade...")
            return
        self.evaluate_positions(btc_price)

        # Verificar stop-loss y take-profit
        if self.btc_balance > 0 and self.last_trade_price:
            current_profit = (btc_price - self.last_trade_price) / self.last_trade_price
            if current_profit < (self.stop_loss_percentage - 1):
                self.log_message("Stop-loss triggered, selling BTC")
                self.sell_all()
                return
            elif current_profit > (self.take_profit_percentage - 1):
                self.log_message("Take-profit triggered, selling BTC")
                self.sell_all()
                return

        votes = []
        for indicator, calculate_decision in self.indicators.items():
            decision = calculate_decision(self.historical_prices)
            self.update_decision_label(self.decision_labels[indicator], decision)
            if decision != "WAITING" and decision != "NEUTRAL":
                votes.append(decision)

        if not votes:
            self.final_decision_label["text"] = "NEUTRAL"
            self.final_decision_label["foreground"] = "gray"
            return

        buy_votes = votes.count("BUY")
        sell_votes = votes.count("SELL")
        
        # Calculate confidence level
        total_votes = len(votes)
        if total_votes == 0:
            confidence = 0
        else:
            confidence = max(buy_votes, sell_votes) / total_votes * 100

        # Apply risk management
        if self.consecutive_losses >= 3:
            self.log_message("Risk management: Reducing trade size due to consecutive losses")
            confidence *= 0.5

        if self.trade_cooldown > 0:
            self.trade_cooldown -= 1
            self.log_message(f"Trade cooldown: {self.trade_cooldown} periods remaining")
            return

        # Make final decision with confidence threshold
        if confidence >= 60:  # Require at least 60% confidence
            final_decision = "BUY" if buy_votes > sell_votes else "SELL"
            self.final_decision_label["text"] = f"{final_decision} ({confidence:.1f}% confident)"
            self.final_decision_label["foreground"] = "green" if final_decision == "BUY" else "red"

            # Execute trade if conditions are met
            if final_decision == "BUY" and self.balance >= MIN_TRADE_AMOUNT:
                self.buy_btc(btc_price)
            elif final_decision == "SELL" and self.btc_balance > 0:
                self.sell_all()
        else:
            self.final_decision_label["text"] = f"NEUTRAL (Low confidence: {confidence:.1f}%)"
            self.final_decision_label["foreground"] = "gray"
        
        self.last_trade_time = time.time()

    def buy_btc(self, btc_price):
        try:
            if self.balance < MIN_TRADE_AMOUNT:
                self.log_message("Insufficient balance for buying")
                return

            # Comprar solo el 20% del saldo disponible
            amount_to_buy = self.balance * 0.2
            btc_amount = (amount_to_buy / btc_price) * (1 - COMISION)

            # Crear una nueva posición abierta
            new_position = {
                "buy_price": btc_price,
                "btc_amount": btc_amount,
                "buy_time": datetime.now(),
                "status": "open"  # Estado de la posición
            }
            self.open_positions.append(new_position)

            # Actualizar saldos
            self.btc_balance += btc_amount
            self.balance -= amount_to_buy

            self.update_ui()
            self.log_message(f"BUY: {btc_amount:.8f} BTC at ${btc_price:.2f}")

            # Actualizar la lista de posiciones abiertas
            self.update_open_positions_list()

        except Exception as e:
            self.log_message(f"Error during buy: {str(e)}")


    def close_position(self, position, btc_price):
        try:
            # Calcular el monto de la venta
            sale_amount = position["btc_amount"] * btc_price * (1 - COMISION)

            # Actualizar saldos
            self.balance += sale_amount
            self.btc_balance -= position["btc_amount"]

            # Marcar la posición como cerrada
            position["status"] = "closed"
            position["sell_price"] = btc_price
            position["sell_time"] = datetime.now()

            # Calcular el beneficio
            profit_percentage = ((btc_price - position["buy_price"]) / position["buy_price"]) * 100
            self.log_message(f"SELL: {position['btc_amount']:.8f} BTC at ${btc_price:.2f} (Profit: {profit_percentage:.2f}%)")

            # Actualizar la lista de posiciones cerradas
            self.update_closed_positions_list()

            # Registrar el rendimiento después de cerrar la posición
            self.log_performance()

        except Exception as e:
            self.log_message(f"Error during sell: {str(e)}")

       

    def sell_all(self):
        try:
            if self.btc_balance <= 0:
                self.log_message("No BTC available for selling")
                return

            btc_price = self.get_current_btc_price()
            if btc_price is None:
                self.log_message("Could not get current BTC price")
                return

            sale_amount = self.btc_balance * btc_price * (1 - COMISION)
            temp_btc = self.btc_balance

            # Update profit tracking
            if self.last_trade_price:
                profit_percentage = ((btc_price - self.last_trade_price) / self.last_trade_price) * 100
                if profit_percentage < 0:
                    self.consecutive_losses += 1
                    self.trade_cooldown = min(self.consecutive_losses * 5, 20)  # Increasing cooldown with losses
                else:
                    self.consecutive_losses = 0
                    self.trade_cooldown = 0

                self.log_message(f"Trade profit: {profit_percentage:.2f}%")

            self.balance += sale_amount
            self.btc_balance = 0
            self.last_trade_price = None

            self.update_ui()
            self.log_message(f"SELL: {temp_btc:.8f} BTC at ${btc_price:.2f}")

        except Exception as e:
            self.log_message(f"Error during sell: {str(e)}")

    def evaluate_positions(self, btc_price):
        for position in self.open_positions:
            if position["status"] == "open":
                # Evaluar la posición con los jueces
                votes = []
                for indicator, calculate_decision in self.indicators.items():
                    decision = calculate_decision(self.historical_prices)  # Solo pasar los precios históricos
                    if decision != "WAITING" and decision != "NEUTRAL":
                        votes.append(decision)

                # Contar votos
                buy_votes = votes.count("BUY")
                sell_votes = votes.count("SELL")

                # Tomar decisión final
                if sell_votes > buy_votes:
                    self.close_position(position, btc_price)
                elif buy_votes > sell_votes:
                    self.log_message(f"Position at {position['buy_price']} remains open.")
                else:
                    self.log_message(f"Position at {position['buy_price']} is neutral.")

    def close_position(self, position, btc_price):
        try:
            # Calcular el monto de la venta
            sale_amount = position["btc_amount"] * btc_price * (1 - COMISION)

            # Actualizar saldos
            self.balance += sale_amount
            self.btc_balance -= position["btc_amount"]

            # Marcar la posición como cerrada
            position["status"] = "closed"
            position["sell_price"] = btc_price
            position["sell_time"] = datetime.now()

            # Calcular el beneficio
            profit_percentage = ((btc_price - position["buy_price"]) / position["buy_price"]) * 100
            self.log_message(f"SELL: {position['btc_amount']:.8f} BTC at ${btc_price:.2f} (Profit: {profit_percentage:.2f}%)")

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

            # Registrar el rendimiento al finalizar la sesión de trading
            self.log_performance()
             


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
def start_trading(self):
    self.running = True
    self.trade_interval = 60  # Intervalo de 60 segundos entre operaciones
    self.schedule_next_trade()

def schedule_next_trade(self):
    if self.running:
        self.root.after(self.trade_interval * 1000, self.perform_trade)

def perform_trade(self):
    if self.running:
        btc_price = self.get_current_btc_price()
        if btc_price is not None:
            self.make_decision(btc_price)
        self.schedule_next_trade()


    self.log_message(f"Current BTC balance: {self.btc_balance:.8f} BTC")

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