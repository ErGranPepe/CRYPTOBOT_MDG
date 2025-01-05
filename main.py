import requests
import time
import threading
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Configuración de las APIs
BINANCE_API_URL = 'https://api.binance.com/api/v3/ticker/price'
COMISION = 0.001  # 0.1% de comisión por operación
saldo_dinero = 10000  # Capital inicial en USD
saldo_btc = 0  # No tenemos BTC al inicio
precios_historial = {'BTC': []}  # Diccionario para precios históricos por moneda
trading_activo = False  # Estado inicial del trading
crypto_seleccionada = 'BTC'  # Moneda seleccionada para operar
compras_realizadas = []  # Registro de compras realizadas

# Parámetros de estrategia modificados
MARGEN_GANANCIA = 1.002  # 0.2% de ganancia mínima para vender
UMBRAL_CAMBIO_PRECIO = 10  # $10 para el cambio de precio en USD
INTERVALO_ACTUALIZACION = 5  # Intervalo de actualización en segundos

# Funciones principales
def obtener_precio_binance(cripto='BTC'):
    try:
        response = requests.get(f'{BINANCE_API_URL}?symbol={cripto}USDT').json()
        return float(response['price'])
    except Exception as e:
        print(f"Error al obtener datos de Binance: {e}")
        return None

def obtener_precio(cripto='BTC'):
    precio = obtener_precio_binance(cripto)
    if precio is not None:
        return precio
    else:
        return precios_historial[cripto][-1] if precios_historial[cripto] else 20000

def inicializar_precios():
    precio_inicial = obtener_precio_binance(crypto_seleccionada)
    if precio_inicial:
        precios_historial[crypto_seleccionada].append(precio_inicial)
        return precio_inicial
    else:
        print("Error al obtener el precio inicial. Usando valor predeterminado.")
        precios_historial[crypto_seleccionada].append(20000)
        return 20000

def actualizar_precios():
    global precios_historial
    precio_actual = obtener_precio(crypto_seleccionada)
    precios_historial[crypto_seleccionada].append(precio_actual)
    lbl_precio.config(text=f"Precio actual {crypto_seleccionada}: ${precio_actual:.2f}")
    lbl_saldo.config(text=f"Saldo: ${saldo_dinero:.2f} | {crypto_seleccionada}: {saldo_btc:.6f}")
    print(f"Actualización: Precio {crypto_seleccionada}: ${precio_actual:.2f} | Saldo: ${saldo_dinero:.2f} | {crypto_seleccionada}: {saldo_btc:.6f}")
    return precio_actual

def realizar_compra(precio):
    global saldo_dinero, saldo_btc, compras_realizadas
    cantidad_comprada = min((saldo_dinero * 0.2) / precio, saldo_dinero / precio)  # Usar máximo 20% del saldo
    cantidad_comprada *= (1 - COMISION)
    costo_total = precio * cantidad_comprada
    if saldo_dinero >= costo_total:
        saldo_dinero -= costo_total
        saldo_btc += cantidad_comprada
        compras_realizadas.append({'precio': precio, 'cantidad': cantidad_comprada})
        print(f"Compra realizada: {cantidad_comprada:.6f} BTC a ${precio:.2f}")
        return True
    return False

def realizar_venta(precio):
    global saldo_dinero, saldo_btc, compras_realizadas
    if saldo_btc > 0:
        cantidad_vendida = saldo_btc * 0.2  # Vender el 20% de lo que se tiene
        cantidad_vendida *= (1 - COMISION)
        ganancia_total = precio * cantidad_vendida
        saldo_btc -= cantidad_vendida
        saldo_dinero += ganancia_total
        print(f"Venta realizada: {cantidad_vendida:.6f} BTC a ${precio:.2f}")
        return True
    return False

def estrategia_trading(precio_actual):
    if len(precios_historial[crypto_seleccionada]) < 10:
        return

    media_corta = sum(precios_historial[crypto_seleccionada][-5:]) / 5
    media_larga = sum(precios_historial[crypto_seleccionada][-10:]) / 10
    
    print(f"Precio actual: {precio_actual:.2f} | Media corta: {media_corta:.2f} | Media larga: {media_larga:.2f}")

    if precio_actual < media_larga - UMBRAL_CAMBIO_PRECIO and saldo_dinero > 10:
        if realizar_compra(precio_actual):
            print(f"Estrategia: Compra ejecutada a ${precio_actual:.2f}")
    elif precio_actual > media_corta + UMBRAL_CAMBIO_PRECIO and saldo_btc > 0:
        if realizar_venta(precio_actual):
            print(f"Estrategia: Venta ejecutada a ${precio_actual:.2f}")
    else:
        print(f"No se cumplieron condiciones de compra/venta. Diferencia con media corta: ${abs(precio_actual - media_corta):.2f}")

def actualizar_graficos():
    ax_precio.clear()
    ax_precio.plot(precios_historial[crypto_seleccionada], label=f"{crypto_seleccionada} Precio", linewidth=2, color='blue')
    ax_precio.set_title("Histórico de Precios")
    ax_precio.set_xlabel("Transacciones")
    ax_precio.set_ylabel("Precio ($)")
    ax_precio.legend()
    canvas.draw()

def iniciar_trading():
    global trading_activo
    trading_activo = True
    while trading_activo:
        try:
            precio_actual = actualizar_precios()
            estrategia_trading(precio_actual)
            actualizar_graficos()
            time.sleep(INTERVALO_ACTUALIZACION)
        except Exception as e:
            print(f"Error en el trading: {e}")
            break

def pausar_reanudar():
    global trading_activo
    if trading_activo:
        trading_activo = False
        btn_pausar.config(text="Reanudar Trading")
    else:
        btn_pausar.config(text="Pausar Trading")
        threading.Thread(target=iniciar_trading, daemon=True).start()

# Creación de la ventana principal
ventana = tk.Tk()
ventana.title("Simulación de Trading de Criptomonedas")
ventana.geometry("800x600")
ventana.config(bg="#2e3b4e")

# Inicialización
inicializar_precios()

# Etiquetas
lbl_precio = tk.Label(ventana, text="Precio actual BTC: $0.00", font=("Arial", 14), fg="white", bg="#2e3b4e")
lbl_precio.pack(pady=10)

lbl_saldo = tk.Label(ventana, text=f"Saldo: ${saldo_dinero:.2f} | BTC: {saldo_btc:.6f}", font=("Arial", 14), fg="white", bg="#2e3b4e")
lbl_saldo.pack(pady=10)

# Botones
btn_pausar = tk.Button(ventana, text="Iniciar Trading", command=pausar_reanudar, font=("Arial", 12), bg="#5a6f8f", fg="white")
btn_pausar.pack(pady=10)

# Gráficos
figura = plt.Figure(figsize=(8, 4), dpi=100)
ax_precio = figura.add_subplot(111)
canvas = FigureCanvasTkAgg(figura, ventana)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Iniciar el bucle de la ventana
ventana.mainloop()
