import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import requests
import threading
import time
import numpy as np

BINANCE_API_URL = 'https://api.binance.com/api/v3/ticker/price'
COMISION = 0.001  # Comisión del 0.1%
INITIAL_BALANCE = 1000  # Saldo inicial en USD

class CryptoBot:
    def __init__(self, root):
        self.balance = INITIAL_BALANCE
        self.btc_balance = 0
        self.starting_balance = INITIAL_BALANCE
        self.historical_prices = []
        self.running = False

        self.create_ui(root)

    def create_ui(self, root):
        root.title("Crypto Auto-Trading Bot")

        # Frame para la cuenta
        account_frame = ttk.LabelFrame(root, text="Cuenta", padding=10)
        account_frame.pack(fill="x")

        ttk.Label(account_frame, text="Saldo en USD:").grid(row=0, column=0, sticky="w")
        self.usd_balance_label = ttk.Label(account_frame, text=f"{self.balance:.2f}")
        self.usd_balance_label.grid(row=0, column=1, sticky="e")

        ttk.Label(account_frame, text="Saldo en BTC:").grid(row=1, column=0, sticky="w")
        self.btc_balance_label = ttk.Label(account_frame, text=f"{self.btc_balance:.8f}")
        self.btc_balance_label.grid(row=1, column=1, sticky="e")

        ttk.Label(account_frame, text="Beneficio:").grid(row=2, column=0, sticky="w")
        self.profit_label = ttk.Label(account_frame, text=f"0.00")
        self.profit_label.grid(row=2, column=1, sticky="e")

        # Frame para acciones
        actions_frame = ttk.Frame(root, padding=10)
        actions_frame.pack(fill="x")

        self.start_button = ttk.Button(actions_frame, text="Iniciar Bot", command=self.start_bot)
        self.start_button.grid(row=0, column=0, padx=5)

        self.pause_button = ttk.Button(actions_frame, text="Pausar Bot", command=self.pause_bot, state="disabled")
        self.pause_button.grid(row=0, column=1, padx=5)

        self.sell_button = ttk.Button(actions_frame, text="Vender Todo", command=self.sell_all, state="disabled")
        self.sell_button.grid(row=0, column=2, padx=5)

        self.exit_button = ttk.Button(actions_frame, text="Salir", command=root.quit)
        self.exit_button.grid(row=0, column=3, padx=5)

        # Frame para datos en vivo
        live_data_frame = ttk.LabelFrame(root, text="Datos en Vivo", padding=10)
        live_data_frame.pack(fill="both", expand=True)

        ttk.Label(live_data_frame, text="Precio BTC/USDT:").grid(row=0, column=0, sticky="w")
        self.btc_price_label = ttk.Label(live_data_frame, text="Cargando...")
        self.btc_price_label.grid(row=0, column=1, sticky="e")

        # Frame para gráficos
        graph_frame = ttk.LabelFrame(root, text="Indicadores y Decisiones", padding=10)
        graph_frame.pack(fill="both", expand=True)

        self.figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Precio BTC y Indicadores")
        self.ax.set_xlabel("Tiempo")
        self.ax.set_ylabel("Precio")
        self.canvas = FigureCanvasTkAgg(self.figure, graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Frame para decisiones
        decision_frame = ttk.LabelFrame(root, text="Decisiones de Jueces", padding=10)
        decision_frame.pack(fill="both", expand=True)

        self.decision_labels = {}
        indicators = ["MA", "RSI", "Bollinger", "MACD", "Estocástico", "Fibonacci", "Ichimoku", "Desviación", "Aroon"]
        for i, indicator in enumerate(indicators):
            ttk.Label(decision_frame, text=f"Decisión de {indicator}:").grid(row=i, column=0, sticky="w")
            self.decision_labels[indicator] = ttk.Label(decision_frame, text="Esperando...", foreground="gray")
            self.decision_labels[indicator].grid(row=i, column=1, sticky="e")

        ttk.Label(decision_frame, text="Decisión Final del Juez Supremo:").grid(row=len(indicators), column=0, sticky="w")
        self.final_decision_label = ttk.Label(decision_frame, text="Esperando...", foreground="gray")
        self.final_decision_label.grid(row=len(indicators), column=1, sticky="e")

    def start_bot(self):
        self.running = True
        self.start_button["state"] = "disabled"
        self.pause_button["state"] = "normal"
        self.sell_button["state"] = "normal"
        self.bot_thread = threading.Thread(target=self.run_bot)
        self.bot_thread.start()

    def pause_bot(self):
        self.running = False
        self.start_button["state"] = "normal"
        self.pause_button["state"] = "disabled"

    def sell_all(self):
        if self.btc_balance > 0:
            btc_price = self.get_current_btc_price()
            self.balance += (1 - COMISION) * self.btc_balance * btc_price
            self.btc_balance = 0
            self.update_ui()

    def run_bot(self):
        while self.running:
            btc_price = self.get_current_btc_price()
            if btc_price:
                self.historical_prices.append(btc_price)
                if len(self.historical_prices) > 200:
                    self.historical_prices.pop(0)
                self.update_graph()
                self.make_decision(btc_price)
                self.update_ui()
            time.sleep(1)

    def get_current_btc_price(self):
        try:
            response = requests.get(BINANCE_API_URL, params={"symbol": "BTCUSDT"})
            response.raise_for_status()
            data = response.json()
            return float(data["price"])
        except Exception as e:
            print(f"Error al obtener el precio BTC/USDT: {e}")
            return None

    def make_decision(self, btc_price):
        if len(self.historical_prices) < 50:
            for label in self.decision_labels.values():
                label["text"] = "Esperando..."
                label["foreground"] = "gray"
            self.final_decision_label["text"] = "Esperando..."
            self.final_decision_label["foreground"] = "gray"
            return

        votes = []

        # Indicadores y decisiones
        # (El código para los indicadores se mantiene igual)

        # Juez Supremo - Optimización de beneficios
        buy_votes = votes.count("COMPRAR")
        sell_votes = votes.count("VENDER")
        hold_votes = votes.count("MANTENER")
        final_decision = "COMPRAR" if buy_votes > sell_votes else ("VENDER" if sell_votes > buy_votes else "MANTENER")

        # Optimización basada en ganancias potenciales
        if final_decision == "COMPRAR" and self.balance > 0:
            self.btc_balance += (1 - COMISION) * self.balance / btc_price
            self.balance = 0
        elif final_decision == "VENDER" and self.btc_balance > 0:
            self.balance += (1 - COMISION) * self.btc_balance * btc_price
            self.btc_balance = 0

        self.final_decision_label["text"] = final_decision
        self.final_decision_label["foreground"] = "green" if final_decision == "COMPRAR" else ("red" if final_decision == "VENDER" else "gray")

    def update_ui(self):
        self.usd_balance_label["text"] = f"{self.balance:.2f}"
        self.btc_balance_label["text"] = f"{self.btc_balance:.8f}"
        profit = ((self.balance + self.btc_balance * self.get_current_btc_price()) - self.starting_balance) / self.starting_balance * 100
        self.profit_label["text"] = f"{profit:.2f}%"
        self.profit_label["foreground"] = "green" if profit >= 0 else "red"

    def update_graph(self):
        self.ax.clear()
        self.ax.plot(self.historical_prices, label="Precio BTC")
        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    bot = CryptoBot(root)
    root.mainloop()
