import requests
import time
import threading

# Variables globales
COMISION = 0.001  # 0.1% de comisión por operación
saldo_dinero = 10000  # Capital inicial en USD
saldo_btc = 0  # No tenemos BTC al inicio
precios_historial = {'BTC': []}  # Diccionario para precios históricos por moneda
inversion_inicial = 0  # Inicializamos la inversión realizada
trading_activo = True  # Estado del trading
crypto_seleccionada = 'BTC'  # Moneda seleccionada para operar (BTC por defecto)
inversion_total = 0  # Variable que guarda el valor total invertido hasta el momento

# Configuración de las APIs
BINANCE_API_URL = 'https://api.binance.com/api/v3/ticker/price'
COINMARKETCAP_API_URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
HEADERS_COINMARKETCAP = {
    'X-CMC_PRO_API_KEY': 'tu_api_key',  # Reemplaza con tu API Key de CoinMarketCap
    'Accepts': 'application/json'
}

# Función para obtener el precio de Binance
def obtener_precio_binance(cripto='BTC'):
    try:
        response = requests.get(f'{BINANCE_API_URL}?symbol={cripto}USDT').json()
        return float(response['price'])
    except Exception as e:
        print(f"Error al obtener datos de Binance: {e}")
        return precios_historial.get(cripto, [-1])[-1]  # Último precio disponible si falla

# Función para obtener el precio de CoinMarketCap
def obtener_precio_coinmarketcap(cripto='BTC'):
    try:
        params = {'symbol': cripto, 'convert': 'USD', 'limit': '1'}
        response = requests.get(COINMARKETCAP_API_URL, headers=HEADERS_COINMARKETCAP, params=params).json()
        if 'data' in response and len(response['data']) > 0:
            return float(response['data'][0]['quote']['USD']['price'])
        else:
            return precios_historial.get(cripto, [20000])[-1]
    except Exception as e:
        print(f"Error al obtener datos de CoinMarketCap: {e}")
        return precios_historial.get(cripto, [20000])[-1]

# Función para obtener el precio promedio de BTC
def obtener_precio(cripto='BTC'):
    precio_binance = obtener_precio_binance(cripto)
    precio_coinmarketcap = obtener_precio_coinmarketcap(cripto)
    if precio_binance == -1 and precio_coinmarketcap == -1:
        return precios_historial.get(cripto, [20000])[-1]
    return (precio_binance + precio_coinmarketcap) / 2  # Promedio de las dos fuentes

# Función para calcular el cambio porcentual
def calcular_cambio_porcentual(precio_anterior, precio_actual):
    return (precio_actual - precio_anterior) / precio_anterior * 100

# Función para realizar la compra de BTC
def comprar_btc(saldo, precio_actual, cantidad):
    total_compra = cantidad * precio_actual
    comision_compra = total_compra * COMISION
    if saldo >= total_compra + comision_compra:
        saldo -= total_compra + comision_compra
        inversion_realizada = total_compra
        print(f"Compra: {cantidad:.6f} BTC a ${precio_actual:.2f} (Comisión: ${comision_compra:.2f})")
        return saldo, cantidad, inversion_realizada
    else:
        print(f"Error: No tienes suficiente saldo para comprar. Necesitas ${total_compra + comision_compra:.2f}, pero tienes ${saldo:.2f}.")
        return saldo, 0, 0

# Función para realizar la venta de BTC
def vender_btc(saldo_btc, saldo_dinero, precio_actual, cantidad, inversion_realizada):
    if saldo_btc >= cantidad:
        total_venta = cantidad * precio_actual
        comision_venta = total_venta * COMISION
        saldo_dinero += total_venta - comision_venta
        ganancia_neta = total_venta - inversion_realizada - comision_venta
        print(f"Venta: {cantidad:.6f} BTC a ${precio_actual:.2f} (Comisión: ${comision_venta:.2f}) - Ganancia Neta: ${ganancia_neta:.2f}")
        return saldo_btc - cantidad, saldo_dinero, ganancia_neta
    else:
        print("Error: No tienes suficiente BTC para vender.")
        return saldo_btc, saldo_dinero, 0

# Estrategia de trading
def estrategia_trading(precios_historial, saldo, saldo_btc):
    if len(precios_historial[crypto_seleccionada]) < 2:
        return saldo, saldo_btc, 0

    precio_actual = precios_historial[crypto_seleccionada][-1]
    precio_anterior = precios_historial[crypto_seleccionada][-2]

    cambio_porcentual = calcular_cambio_porcentual(precio_anterior, precio_actual)

    if cambio_porcentual > 0.1:
        cantidad_btc = saldo / precio_actual * 0.1  # Compra el 10% del saldo disponible
        nuevo_saldo, cantidad_comprada, inversion_realizada = comprar_btc(saldo, precio_actual, cantidad_btc)
        return nuevo_saldo, cantidad_comprada + saldo_btc, inversion_realizada
    elif cambio_porcentual < -0.1 and saldo_btc > 0:
        nuevo_saldo_btc, nuevo_saldo_dinero, ganancia_neta = vender_btc(saldo_btc, saldo, precio_actual, saldo_btc, inversion_inicial)
        return nuevo_saldo_btc, nuevo_saldo_dinero, ganancia_neta

    return saldo, saldo_btc, 0

# Función principal de trading
def iniciar_trading():
    global precios_historial, saldo_dinero, saldo_btc, inversion_inicial, trading_activo, crypto_seleccionada, inversion_total
    while trading_activo:
        try:
            precio_actual = obtener_precio(crypto_seleccionada)
            precios_historial.setdefault(crypto_seleccionada, []).append(precio_actual)

            print(f"Precio actual de {crypto_seleccionada}: ${precio_actual:.2f}")
            print(f"Saldo disponible: ${saldo_dinero:.2f} | BTC en cartera: {saldo_btc:.6f}")

            saldo_dinero, saldo_btc, ganancia_neta = estrategia_trading(precios_historial, saldo_dinero, saldo_btc)

            # Mostrar ganancias netas (esto puede ser parte de un registro)
            print(f"Ganancia Neta: ${ganancia_neta:.2f}")

            time.sleep(10)  # Esperar 10 segundos antes de la siguiente transacción
        except Exception as e:
            print(f"Error durante el trading: {str(e)}")
            break

# Iniciar el trading en un hilo separado
threading.Thread(target=iniciar_trading, daemon=True).start()
