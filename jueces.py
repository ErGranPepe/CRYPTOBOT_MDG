import requests
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt


BINANCE_API_URL = 'https://api.binance.com/api/v3/ticker/price'
COMISION = 0.001
INTERVALO_ACTUALIZACION = 10  # en segundos

# Estado del mercado y balances
saldo_dinero = 10000  # saldo en USDT
saldo_btc = 0  # saldo en BTC
precio_promedio_compra = 0
precios_historial = {'BTC': []}
trading_activo = False
crypto_seleccionada = 'BTC'


class Juez:
    def __init__(self, nombre, indicador, ponderacion=1):
        self.nombre = nombre
        self.indicador = indicador
        self.voto = {'accion': 'MANTENER', 'confianza': 5}
        self.ponderacion = ponderacion

    def votar(self, precio_actual, precios_historicos, saldo_dinero, saldo_btc, precio_promedio_compra):
        self.voto = self.indicador(precio_actual, precios_historicos, saldo_dinero, saldo_btc, precio_promedio_compra)
        self.voto['confianza'] *= self.ponderacion 
        return self.voto


def media_movil(precio_actual, precios_historicos, saldo_dinero, saldo_btc, precio_promedio_compra, periodo=10):
    if len(precios_historicos) < periodo:
        return {'accion': 'MANTENER', 'confianza': 5}
    media = sum(precios_historicos[-periodo:]) / periodo
    diferencia = (precio_actual - media) / media
    confianza = max(1, min(10, int(5 + diferencia * 100)))
    
    if precio_actual > media * 1.02 and saldo_dinero > 0:
        return {'accion': 'COMPRAR', 'confianza': confianza}
    elif precio_actual < media * 0.98 and saldo_btc > 0:
        return {'accion': 'VENDER', 'confianza': confianza}
    else:
        return {'accion': 'MANTENER', 'confianza': confianza}

def rsi(precio_actual, precios_historicos, saldo_dinero, saldo_btc, precio_promedio_compra, periodo=14):
    if len(precios_historicos) < periodo + 1:
        return {'accion': 'MANTENER', 'confianza': 5}
    cambios = np.diff(precios_historicos[-periodo-1:])
    ganancias = np.where(cambios > 0, cambios, 0)
    perdidas = np.where(cambios < 0, -cambios, 0)
    media_ganancias = np.mean(ganancias)
    media_perdidas = np.mean(perdidas)
    if media_perdidas == 0:
        return {'accion': 'COMPRAR', 'confianza': 10}
    rs = media_ganancias / media_perdidas
    rsi = 100 - (100 / (1 + rs))
    confianza = max(1, min(10, int(rsi / 10)))
    
    if rsi < 30 and saldo_dinero > 0:
        return {'accion': 'COMPRAR', 'confianza': confianza}
    elif rsi > 70 and saldo_btc > 0:
        return {'accion': 'VENDER', 'confianza': confianza}
    else:
        return {'accion': 'MANTENER', 'confianza': confianza}

def tendencia(precio_actual, precios_historicos, saldo_dinero, saldo_btc, precio_promedio_compra, periodo=20):
    if len(precios_historicos) < periodo:
        return {'accion': 'MANTENER', 'confianza': 5}
    tendencia = (precio_actual - precios_historicos[-periodo]) / precios_historicos[-periodo]
    confianza = max(1, min(10, int(5 + tendencia * 100)))
    
    if tendencia > 0.05 and saldo_dinero > 0:
        return {'accion': 'COMPRAR', 'confianza': confianza}
    elif tendencia < -0.05 and saldo_btc > 0:
        return {'accion': 'VENDER', 'confianza': confianza}
    else:
        return {'accion': 'MANTENER', 'confianza': confianza}

# Jueces con ponderaciones más agresivas
jueces = [
    Juez("Media Móvil", media_movil, ponderacion=2),
    Juez("RSI", rsi, ponderacion=1.5),
    Juez("Tendencia", tendencia, ponderacion=1.5)
]

# Obtener el precio desde Binance
def obtener_precio_binance(cripto='BTC'):
    try:
        response = requests.get(f'{BINANCE_API_URL}?symbol={cripto}USDT')
        response.raise_for_status()  
        data = response.json()
        if 'price' in data:
            return float(data['price'])
        else:
            print("Error: 'price' no encontrado en la respuesta de Binance")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error al obtener datos de Binance: {e}")
        return None

def obtener_precio(cripto='BTC'):
    precio = obtener_precio_binance(cripto)
    if precio is not None:
        return precio
    else:
        return precios_historial[cripto][-1] if precios_historial[cripto] else 20000

# Funciones de compra y venta
def realizar_compra(precio, cantidad):
    global saldo_dinero, saldo_btc, precio_promedio_compra
    costo_total = precio * cantidad * (1 + COMISION)
    if saldo_dinero >= costo_total:
        saldo_dinero -= costo_total
        saldo_btc += cantidad
        precio_promedio_compra = (precio_promedio_compra * saldo_btc + precio * cantidad) / (saldo_btc + cantidad)
        print(f"Compra realizada: {cantidad:.6f} BTC a ${precio:.2f}")
        return True
    return False

def realizar_venta(precio, cantidad):
    global saldo_dinero, saldo_btc
    if saldo_btc >= cantidad:
        ganancia_total = precio * cantidad * (1 - COMISION)
        saldo_btc -= cantidad
        saldo_dinero += ganancia_total
        print(f"Venta realizada: {cantidad:.6f} BTC a ${precio:.2f}")
        return True
    return False

# Juez Supremo: toma decisiones ponderadas
def juez_supremo(votos):
    acciones = {'COMPRAR': 0, 'VENDER': 0, 'MANTENER': 0}
    confianza_total = 0
    for voto in votos:
        acciones[voto['accion']] += voto['confianza']
        confianza_total += voto['confianza']
    
    accion_final = max(acciones, key=acciones.get)
    confianza_final = acciones[accion_final] / confianza_total
    
    if confianza_final > 6:
        return {'accion': accion_final, 'confianza': confianza_final}
    else:
        return {'accion': 'MANTENER', 'confianza': 5}


def estrategia_trading(precio_actual):
    votos = []
    for juez in jueces:
        voto = juez.votar(precio_actual, precios_historial[crypto_seleccionada], saldo_dinero, saldo_btc, precio_promedio_compra)
        votos.append(voto)

    decision_final = juez_supremo(votos)
    print(f"Decisión final del Juez Supremo: {decision_final['accion']} con confianza {decision_final['confianza']:.2f}")

    if decision_final['accion'] == 'COMPRAR' and saldo_dinero > 0:
        cantidad = min(saldo_dinero / precio_actual * 0.1, saldo_dinero / precio_actual)
        realizar_compra(precio_actual, cantidad)
    elif decision_final['accion'] == 'VENDER' and saldo_btc > 0:
        cantidad = saldo_btc * 0.1
        realizar_venta(precio_actual, cantidad)

    actualizar_interfaz_jueces(votos, decision_final)

def actualizar_interfaz_jueces(votos, decision_final):
    for i, juez in enumerate(jueces):
        tabla_votos.set(i, "Acción", votos[i]['accion'])
        tabla_votos.set(i, "Confianza", f"{votos[i]['confianza']:.2f}")
    
    supremo_iid = len(jueces)  # Siempre la última fila
    if not tabla_votos.exists(supremo_iid):  # Si no existe, insertamos
        tabla_votos.insert('', 'end', iid=supremo_iid, values=("Juez Supremo", decision_final['accion'], f"{decision_final['confianza']:.2f}"))
    else:  # Si ya existe, actualizamos
        tabla_votos.set(supremo_iid, "Acción", decision_final['accion'])
        tabla_votos.set(supremo_iid, "Confianza", f"{decision_final['confianza']:.2f}")


def actualizar_datos():
    if trading_activo:
        precio_actual = obtener_precio(crypto_seleccionada)
        if precio_actual is not None:
            precios_historial[crypto_seleccionada].append(precio_actual)
            estrategia_trading(precio_actual)
        else:
            print("No se pudo obtener el precio de BTC")
    ventana.after(INTERVALO_ACTUALIZACION * 1000, actualizar_datos)

# Interfaz gráfica
ventana = tk.Tk()
ventana.title("Simulador de Trading con Jueces")
ventana.geometry("800x600")

# Etiquetas
lbl_precio = tk.Label(ventana, text="Precio actual BTC: $0.00")
lbl_precio.pack()
lbl_saldo = tk.Label(ventana, text="Saldo: $0.00 | BTC: 0.000000")
lbl_saldo.pack()
lbl_promedio = tk.Label(ventana, text="Precio promedio de compra: $0.00")
lbl_promedio.pack()
lbl_beneficio = tk.Label(ventana, text="Beneficio actual: $0.00")
lbl_beneficio.pack()

# Tabla de votaciones
tabla_votos = ttk.Treeview(ventana, columns=("Juez", "Acción", "Confianza"), show="headings")
tabla_votos.heading("Juez", text="Juez")
tabla_votos.heading("Acción", text="Acción")
tabla_votos.heading("Confianza", text="Confianza")
tabla_votos.pack()


for juez in jueces:
    tabla_votos.insert('', 'end', iid=jueces.index(juez), values=(juez.nombre, 'MANTENER', '5.00'))


def iniciar_trading():
    global trading_activo
    trading_activo = True
    actualizar_datos()

boton_iniciar = tk.Button(ventana, text="Iniciar Trading", command=iniciar_trading)
boton_iniciar.pack()

ventana.mainloop()
