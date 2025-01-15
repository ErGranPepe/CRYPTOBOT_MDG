import random
import logging

# Configuración de logging para ver los resultados
logging.basicConfig(level=logging.INFO)

# Clase que representa una Sinapsis
class Sinapsis:
    def __init__(self, neurona_destino, peso=0.5):
        self.neurona_destino = neurona_destino
        self.peso = peso
        self.potencial = 0  # Estímulo recibido por la sinapsis

# Clase que representa una Neurona
class Neurona:
    def __init__(self, id, umbral_activacion=-55, resistencia_membrana=10):
        self.id = id
        self.potencial_membrana = -70  # Valor inicial de reposo
        self.umbral_activacion = umbral_activacion
        self.resistencia_membrana = resistencia_membrana
        self.conexiones = []  # Lista de conexiones entrantes

    def agregar_conexion(self, sinapsis):
        """Agrega una conexión a la neurona"""
        self.conexiones.append(sinapsis)

    def calcular_potencial(self):
        """Calcula el potencial de membrana sumando los estímulos recibidos"""
        entrada_total = sum([sinapsis.potencial for sinapsis in self.conexiones])
        self.potencial_membrana += entrada_total / self.resistencia_membrana

    def disparar(self):
        """Dispara la neurona si su potencial es mayor que el umbral"""
        self.calcular_potencial()
        if self.potencial_membrana >= self.umbral_activacion:
            self.potencial_membrana = -70  # Resetear después del disparo
            return True  # La neurona dispara
        return False  # La neurona no dispara

# Clase que representa una región cerebral (como la corteza, el cerebelo, etc.)
class RegionCerebral:
    def __init__(self, nombre):
        self.nombre = nombre
        self.neuronas = []

    def agregar_neurona(self, neurona):
        """Agrega una neurona a la región cerebral"""
        self.neuronas.append(neurona)

    def actividad_global(self, tiempo_actual):
        """Revisa la actividad de las neuronas y si disparan"""
        actividad_actual = 0
        for neurona in self.neuronas:
            if random.random() < 0.9:  # Probabilidad de activación (90%)
                if neurona.disparar():
                    actividad_actual += 1
        logging.info(f"Actividad en {self.nombre}: {actividad_actual} neuronas activas.")

# Crear las regiones cerebrales (como corteza, cerebelo, etc.)
corteza = RegionCerebral("Corteza")
cerebelo = RegionCerebral("Cerebelo")
hipocampo = RegionCerebral("Hipocampo")
ganglios_basales = RegionCerebral("Ganglios Basales")
amigdala = RegionCerebral("Amígdala")
tronco_encefalico = RegionCerebral("Tronco Encefálico")

# Crear algunas neuronas y agregarlas a las regiones
for i in range(10):  # Crear 10 neuronas para cada región
    neurona_corteza = Neurona(i + 1)
    corteza.agregar_neurona(neurona_corteza)
    
    neurona_cerebelo = Neurona(i + 11)
    cerebelo.agregar_neurona(neurona_cerebelo)

    neurona_hipocampo = Neurona(i + 21)
    hipocampo.agregar_neurona(neurona_hipocampo)

    neurona_ganglios_basales = Neurona(i + 31)
    ganglios_basales.agregar_neurona(neurona_ganglios_basales)

    neurona_amigdala = Neurona(i + 41)
    amigdala.agregar_neurona(neurona_amigdala)

    neurona_tronco_encefalico = Neurona(i + 51)
    tronco_encefalico.agregar_neurona(neurona_tronco_encefalico)

# Conectar las neuronas entre las regiones
for region in [corteza, cerebelo, hipocampo, ganglios_basales, amigdala, tronco_encefalico]:
    for neurona in region.neuronas:
        for otra_region in [corteza, cerebelo, hipocampo, ganglios_basales, amigdala, tronco_encefalico]:
            if region != otra_region:  # No se conecta la neurona consigo misma
                for otra_neurona in otra_region.neuronas:
                    sinapsis = Sinapsis(otra_neurona, peso=random.uniform(0.1, 1.0))
                    neurona.agregar_conexion(sinapsis)

# Añadir un estímulo inicial a las neuronas
for region in [corteza, cerebelo, hipocampo, ganglios_basales, amigdala, tronco_encefalico]:
    for neurona in region.neuronas:
        neurona.potencial_membrana = -55  # Cerca del umbral de activación

# Simulación durante un ciclo de tiempo
max_tiempo = 100  # Número de ciclos de simulación
for t in range(0, max_tiempo):
    corteza.actividad_global(t)
    cerebelo.actividad_global(t)
    hipocampo.actividad_global(t)
    ganglios_basales.actividad_global(t)
    amigdala.actividad_global(t)
    tronco_encefalico.actividad_global(t)

import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la simulación
num_neuronas = 100  # Número de neuronas
umbral_activacion = 0.5  # Umbral de activación
tiempo_total = 100  # Número de pasos de simulación
probabilidad_conexion = 0.1  # Probabilidad de conexión entre neuronas
conexiones = np.random.rand(num_neuronas, num_neuronas) < probabilidad_conexion  # Matriz de conexiones

# Inicialización del estado de las neuronas
estado_neuronas = np.zeros(num_neuronas)  # Estado inicial (todas inactivas)

# Función para actualizar el estado de las neuronas
def actualizar_estado(estado_neuronas, conexiones):
    # Calcular la suma de entradas de cada neurona
    entradas = np.dot(conexiones, estado_neuronas)
    
    # Activar neuronas si la entrada supera el umbral
    nuevo_estado = (entradas > umbral_activacion).astype(int)
    return nuevo_estado

# Simulación
historial_estados = []

for t in range(tiempo_total):
    # Actualizar el estado de las neuronas
    estado_neuronas = actualizar_estado(estado_neuronas, conexiones)
    
    # Guardar el estado actual de las neuronas
    historial_estados.append(estado_neuronas.copy())
    
    # Imprimir el estado de las neuronas (solo las primeras 10)
    if t % 10 == 0:
        print(f"Paso de tiempo {t}: {estado_neuronas[:10]}...")

# Convertir el historial de estados en una matriz para graficar
historial_estados = np.array(historial_estados)

# Graficar la evolución de la actividad neuronal
plt.figure(figsize=(10, 6))
plt.imshow(historial_estados.T, aspect='auto', cmap='binary')
plt.title('Evolución de la actividad neuronal')
plt.xlabel('Tiempo')
plt.ylabel('Neuronas')
plt.colorbar(label='Actividad')
plt.show()

