import requests
import time
import threading
import tkinter as tk
from tkinter import simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from datetime import datetime

# API de CoinMarketCap
HEADERS = {
    'X-CMC_PRO_API_KEY': '13bc52c5-0281-41bb-ba61-e2feb89acab1',  # Reemplaza con tu API Key
    'Accepts': 'application/json'
}

PARAMS = {
    'start': '1',
    'limit': '10',  # Puedes ajustar el límite según las monedas que necesites
    'convert': 'USD'
}

URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'

# Variables globales
COMISION = 0.001  # 0.1% de comisión por operación
saldo_dinero = 10000  # Capital inicial en USD
saldo_btc = 0  # No tenemos BTC al inicio
precios_historial = []  # Inicializamos el historial vacío
inversion_inicial = 0  # Inicializamos la inversión realizada
trading_activo = True  # Estado del trading

def obtener_precio_btc():
    """Obtiene el precio actual de BTC desde CoinMarketCap."""
    try:
        response = requests.get(URL, headers=HEADERS, params=PARAMS).json()
        for coin in response['data']:
            if coin['symbol'] == 'BTC':
                return float(coin['quote']['USD']['price'])
    except Exception as e:
        agregar_log(f"Error al obtener el precio de BTC: {str(e)}")
        return precios_historial[-1] if precios_historial else 20000  # Valor predeterminado

def calcular_cambio_porcentual(precio_anterior, precio_actual):
    """Calcula el cambio porcentual entre dos precios."""
    return (precio_actual - precio_anterior) / precio_anterior * 100

def comprar_btc(saldo, precio_actual, cantidad):
    """Función para simular la compra de BTC."""
    total_compra = cantidad * precio_actual
    comision_compra = total_compra * COMISION
    if saldo >= total_compra + comision_compra:
        saldo -= total_compra + comision_compra
        agregar_log(f"Compra: {cantidad:.6f} BTC a ${precio_actual:.2f} (Comisión: ${comision_compra:.2f})")
        return saldo, cantidad, total_compra
    else:
        agregar_log("Error: No tienes suficiente saldo para comprar.")
        return saldo, 0, 0

def vender_btc(saldo_btc, saldo_dinero, precio_actual, cantidad, inversion_realizada):
    """Función para simular la venta de BTC."""
    if saldo_btc >= cantidad:
        total_venta = cantidad * precio_actual
        comision_venta = total_venta * COMISION
        saldo_dinero += total_venta - comision_venta
        ganancia = total_venta - inversion_realizada - comision_venta
        agregar_log(f"Venta: {cantidad:.6f} BTC a ${precio_actual:.2f} (Comisión: ${comision_venta:.2f}) - Ganancia/Pérdida: ${ganancia:.2f}")
        return saldo_btc - cantidad, saldo_dinero, ganancia
    else:
        agregar_log("Error: No tienes suficiente BTC para vender.")
        return saldo_btc, saldo_dinero, 0

def estrategia_trading(precios_historial, saldo, saldo_btc):
    """Estrategia simple de trading basada en el cambio porcentual."""
    if len(precios_historial) < 10:
        return saldo, saldo_btc, 0
    
    periodos = 10
    ema = sum(precios_historial[-periodos:]) / periodos
    precio_actual = precios_historial[-1]
    precio_anterior = precios_historial[-2]
    
    cambio_porcentual = calcular_cambio_porcentual(precio_anterior, precio_actual)
    
    if cambio_porcentual > 0.2 and precio_actual > ema:
        cantidad_btc = saldo / precio_actual * 0.1  # Compra el 10% del saldo disponible
        nuevo_saldo, cantidad_comprada, inversion_realizada = comprar_btc(saldo, precio_actual, cantidad_btc)
        return nuevo_saldo, cantidad_comprada + saldo_btc, inversion_realizada
    
    elif cambio_porcentual < -0.2 and precio_actual < ema and saldo_btc > 0:
        nuevo_saldo_btc, nuevo_saldo_dinero, ganancia = vender_btc(saldo_btc, saldo, precio_actual, saldo_btc, inversion_inicial)
        return nuevo_saldo_dinero, 0, ganancia
    
    return saldo, saldo_btc, 0

def iniciar_trading():
    """Función principal para la simulación de trading."""
    global precios_historial, saldo_dinero, saldo_btc, inversion_inicial, trading_activo
    
    while trading_activo:
        try:
            # Obtener el precio actual y actualizar el historial
            precio_actual = obtener_precio_btc()
            precios_historial.append(precio_actual)
            
            # Actualizamos la interfaz gráfica con los nuevos datos
            lbl_precio.config(text=f"Precio actual: ${precio_actual:.2f}")
            lbl_saldo.config(text=f"Saldo actual: ${saldo_dinero:.2f} | BTC en cartera: {saldo_btc:.6f}")
            
            # Aplicamos la estrategia de trading
            saldo_dinero, saldo_btc, inversion_inicial = estrategia_trading(precios_historial, saldo_dinero, saldo_btc)
            
            # Actualizamos gráficos
            actualizar_graficos()
            
            # Pausamos entre transacciones para simular un entorno real
            time.sleep(60)  # 1 minuto entre cada llamada a la API
        
        except Exception as e:
            agregar_log(f"Error durante el trading: {str(e)}")
            break

def actualizar_graficos():
    """Función para actualizar gráficos."""
    ax_precio.clear()
    ax_ganancias.clear()
    
    ax_precio.plot(precios_historial, label="Precio BTC")
    ax_precio.set_title("Histórico de Precio BTC")
    ax_precio.set_xlabel("Transacciones")
    ax_precio.set_ylabel("Precio ($)")
    ax_precio.legend()
    
    ganancias = [saldo_dinero + (saldo_btc * p) - inversion_inicial for p in precios_historial]
    
    ax_ganancias.plot(ganancias, label="Ganancia Total")
    ax_ganancias.set_title("Ganancias Totales")
    ax_ganancias.set_xlabel("Transacciones")
    ax_ganancias.set_ylabel("Ganancia ($)")
    ax_ganancias.legend()
    
    canvas.draw()

def agregar_log(mensaje):
    """Función para agregar log al Text widget."""
    tiempo_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_text.insert(tk.END, f"[{tiempo_actual}] {mensaje}\n")
    log_text.see(tk.END)  # Desplazar hacia abajo automáticamente

def editar_importe_inicial():
    """Función para editar el importe inicial.""" 
    global saldo_dinero
    
    nuevo_importe = simpledialog.askfloat("Importe Inicial", "Introduce el nuevo importe inicial:", minvalue=0)
    
    if nuevo_importe is not None:
        saldo_dinero = nuevo_importe
        lbl_saldo.config(text=f"Saldo actual: ${saldo_dinero:.2f} | BTC en cartera: {saldo_btc:.6f}")

def pausar_reanudar():
    """Función para pausar o reanudar trading."""
    global trading_activo
    
    trading_activo = not trading_activo
    
    if trading_activo:
        btn_pausar.config(text="Pausar Trading")
        threading.Thread(target=iniciar_trading, daemon=True).start()
    else:
        btn_pausar.config(text="Reanudar Trading")

# Creación de la ventana principal
ventana = tk.Tk()
ventana.title("Simulación de Trading de BTC")
ventana.geometry("1000x700")

# Configuración inicial de variables y GUI después de crear la ventana principal.
precios_historial.append(obtener_precio_btc())  # Obtener el precio inicial

# Configuración del marco principal
frame_principal = tk.Frame(ventana)
frame_principal.pack(fill=tk.BOTH, expand=True)

# Etiquetas para mostrar información en la GUI
lbl_precio = tk.Label(frame_principal, text="Precio actual: $0.00", font=("Arial", 14))
lbl_precio.pack(pady=10)

lbl_saldo = tk.Label(frame_principal, text=f"Saldo actual: ${saldo_dinero:.2f} | BTC en cartera: {saldo_btc:.6f}", font=("Arial", 14))
lbl_saldo.pack(pady=10)

# Botones para editar el importe inicial y pausar/reanudar trading.
btn_frame = tk.Frame(frame_principal)
btn_frame.pack(pady=10)

btn_editar_importe = tk.Button(btn_frame, text="Editar Importe Inicial", command=editar_importe_inicial)
btn_editar_importe.pack(side=tk.LEFT, padx=5)

btn_pausar = tk.Button(btn_frame, text="Pausar Trading", command=pausar_reanudar)
btn_pausar.pack(side=tk.LEFT, padx=5)

# Configuración del gráfico con Matplotlib.
figura = plt.Figure(figsize=(10, 5), dpi=100)
ax_precio = figura.add_subplot(211)
ax_ganancias = figura.add_subplot(212)

canvas = FigureCanvasTkAgg(figura, frame_principal)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Configuración del log en la parte inferior.
log_frame = tk.Frame(ventana)
log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=10, pady=10)

log_text = tk.Text(log_frame, height=10, width=80)
log_text.pack(expand=True, fill=tk.BOTH)

# Iniciar el trading en un hilo separado al abrir la ventana.
threading.Thread(target=iniciar_trading, daemon=True).start()

# Ejecutar el bucle principal de la interfaz gráfica.
ventana.mainloop()
