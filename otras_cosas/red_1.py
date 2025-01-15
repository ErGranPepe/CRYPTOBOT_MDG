import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Tuple

# ------------------------------------ CONSTANTES -------------------------------------
TIEMPO_SIMULACION = 500
PROBABILIDAD_DISPARO = 0.05
DECAY_RATE = 0.5
FACTOR_LTP = 1.1
VENTANA_TEMPORAL_LTP = 20
NUM_CONEXIONES = 10000
PERIODO_REFRACTARIO = 5 #ms
NUM_NEURONAS_A_GRAFICAR = 5

# ------------------------------------ CLASES BÁSICAS -------------------------------------

class Neurotransmisor:
    def __init__(self, nombre: str, efecto: str):
        self.nombre = nombre
        self.efecto = efecto

class Neurona:
    def __init__(self, tipo: str, forma: str, neurotransmisor: str, potencial_membrana: float = -70, umbral_activacion: float = -55):
        self.tipo = tipo
        self.forma = forma
        self.neurotransmisor = neurotransmisor
        self.potencial_membrana = potencial_membrana
        self.umbral_activacion = np.random.normal(umbral_activacion, 5) # Umbral variable
        self.conexiones_sinapticas: List[Tuple['Neurona', float]] = []
        self.historial_disparos: List[float] = []
        self.ultimo_disparo: float = -np.inf
        self.tiempo_en_refractario = 0

    def disparar(self, tiempo_actual: float) -> bool:
        if self.tiempo_en_refractario > 0:
            self.tiempo_en_refractario -= 1
            return False
        if self.potencial_membrana >= self.umbral_activacion:
            self.potencial_membrana = -70
            self.historial_disparos.append(tiempo_actual)
            self.ultimo_disparo = tiempo_actual
            self.tiempo_en_refractario = PERIODO_REFRACTARIO
            return True
        return False

    def conectar(self, neurona_destino: 'Neurona', peso: float):
        self.conexiones_sinapticas.append((neurona_destino, peso))

    def recibir_estimulo(self, peso_sinaptico: float):
        if self.tiempo_en_refractario == 0: #No recibe estimulos en periodo refractario
            self.potencial_membrana += peso_sinaptico

    def aplicar_plasticidad(self, tiempo_actual: float, neurona_pre: 'Neurona'):
        diferencia_tiempo = self.ultimo_disparo - neurona_pre.ultimo_disparo
        if abs(diferencia_tiempo) < VENTANA_TEMPORAL_LTP:
            for i, (neurona, peso) in enumerate(neurona_pre.conexiones_sinapticas):
                if neurona is self:
                    if diferencia_tiempo > 0: #LTP
                        neurona_pre.conexiones_sinapticas[i] = (neurona, min(peso * FACTOR_LTP, 5)) #Limitar el crecimiento del peso
                    elif diferencia_tiempo < 0: #LTD
                        neurona_pre.conexiones_sinapticas[i] = (neurona, max(peso / FACTOR_LTP, 0.1)) #Limitar la reducción del peso


class NeuronaPiramidal(Neurona):
    def __init__(self, neurotransmisor: str = "glutamato"):
        super().__init__(tipo="piramidal", forma="piramidal", neurotransmisor=neurotransmisor)

class CelulaPurkinje(Neurona):
    def __init__(self, neurotransmisor: str = "GABA"):
        super().__init__(tipo="Purkinje", forma="multipolar", neurotransmisor=neurotransmisor)

class NeuranaGranular(Neurona):
    def __init__(self, neurotransmisor: str = "glutamato"):
        super().__init__(tipo="granular", forma="esférica", neurotransmisor=neurotransmisor)

class InterneuronaGABAergica(Neurona):
    def __init__(self):
        super().__init__(tipo="interneurona", forma="estrellada", neurotransmisor="GABA")

class NeuronaDopaminergica(Neurona):
    def __init__(self):
        super().__init__(tipo="dopaminérgica", forma="multipolar", neurotransmisor="dopamina")

class NeuronaSerotonergica(Neurona):
    def __init__(self):
        super().__init__(tipo="serotoninérgica", forma="multipolar", neurotransmisor="serotonina")

class NeuronaColinergica(Neurona):
    def __init__(self):
        super().__init__(tipo="colinérgica", forma="multipolar", neurotransmisor="acetilcolina")

# ------------------------------------ ESTRUCTURAS CEREBRALES -------------------------------------

class EstructuraCerebral:
    def __init__(self, nombre: str, neuronas: List[Neurona]):
        self.nombre = nombre
        self.neuronas = neuronas

class Corteza(EstructuraCerebral):
    def __init__(self, num_neuronas: int):
        super().__init__("Corteza", [NeuronaPiramidal() for _ in range(num_neuronas)])

class Cerebelo(EstructuraCerebral):
    def __init__(self, num_purkinje: int, num_granular: int):
        neuronas = [CelulaPurkinje() for _ in range(num_purkinje)] + [NeuranaGranular() for _ in range(num_granular)]
        super().__init__("Cerebelo", neuronas)

class Hipocampo(EstructuraCerebral):
    def __init__(self, num_neuronas: int):
        super().__init__("Hipocampo", [NeuronaPiramidal() for _ in range(num_neuronas)])

class GangliosBasales(EstructuraCerebral):
    def __init__(self, num_neuronas: int):
        neuronas = [NeuronaDopaminergica() for _ in range(num_neuronas // 2)] + [InterneuronaGABAergica() for _ in range(num_neuronas // 2)]
        super().__init__("Ganglios Basales", neuronas)

class Amigdala(EstructuraCerebral):
    def __init__(self, num_neuronas: int):
        neuronas = [NeuronaPiramidal() for _ in range(num_neuronas // 2)] + [InterneuronaGABAergica() for _ in range(num_neuronas // 2)]
        super().__init__("Amígdala", neuronas)

class TroncoEncefalico(EstructuraCerebral):
    def __init__(self, num_neuronas: int):
        neuronas = [NeuronaSerotonergica() for _ in range(num_neuronas // 3)] + \
                   [NeuronaColinergica() for _ in range(num_neuronas // 3)] + \
                   [NeuronaDopaminergica() for _ in range(num_neuronas // 3)]
        super().__init__("Tronco Encefálico", neuronas)

# ------------------------------------ SIMULACIÓN DE ACTIVIDAD CEREBRAL -------------------------------------

class OscilacionCerebral:
    def __init__(self, tipo_onda: str, frecuencia: float):
        self.tipo_onda = tipo_onda
        self.frecuencia = frecuencia

    def generar_onda(self, duracion: float, tasa_muestreo: float) -> np.ndarray:
        t = np.linspace(0, duracion, int(duracion * tasa_muestreo), endpoint=False)
        return np.sin(2 * np.pi * self.frecuencia * t)

class SimuladorEEG:
    def __init__(self, duracion: float, tasa_muestreo: float):
        self.duracion = duracion
        self.tasa_muestreo = tasa_muestreo
        self.ondas: Dict[str, OscilacionCerebral] = {
            "delta": OscilacionCerebral("delta", 2),
            "theta": OscilacionCerebral("theta", 6),
            "alpha": OscilacionCerebral("alpha", 10),
            "beta": OscilacionCerebral("beta", 20),
            "gamma": OscilacionCerebral("gamma", 40)
        }

    def simular_eeg(self) -> np.ndarray:
        eeg = np.zeros(int(self.duracion * self.tasa_muestreo))
        for onda in self.ondas.values():
            eeg += onda.generar_onda(self.duracion, self.tasa_muestreo)
        return eeg

    def visualizar_eeg(self, eeg: np.ndarray):
        t = np.linspace(0, self.duracion, len(eeg), endpoint=False)
        plt.figure(figsize=(12, 6))
        plt.plot(t, eeg)
        plt.title("Simulación de EEG")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Amplitud")
        plt.show()

# ------------------------------------ MAIN - SIMULACIÓN

# ... (Código anterior: clases, estructuras, etc.)

# ------------------------------------ MAIN - SIMULACIÓN -------------------------------------

def main():
    # Crear estructuras cerebrales
    corteza = Corteza(1000)
    cerebelo = Cerebelo(500, 1000)
    hipocampo = Hipocampo(200)
    ganglios_basales = GangliosBasales(300)
    amigdala = Amigdala(100)
    tronco_encefalico = TroncoEncefalico(150)

    estructuras = [corteza, cerebelo, hipocampo, ganglios_basales, amigdala, tronco_encefalico]

    # Crear conexiones entre estructuras (más conexiones para mayor actividad)
    num_conexiones = 5000
    # Crear conexiones entre estructuras (más conexiones)
    # Crear conexiones entre estructuras (con distribución gaussiana)
    for _ in range(NUM_CONEXIONES):
        estructura_origen = random.choice(estructuras)
        estructura_destino = random.choice(estructuras)
        if estructura_origen != estructura_destino:
            neurona_origen = random.choice(estructura_origen.neuronas)
            neurona_destino = random.choice(estructura_destino.neuronas)
            peso = np.random.normal(1.0, 0.5) # Distribución gaussiana
            peso = max(0.1, min(peso, 2)) # Limitar el rango del peso
            neurona_origen.conectar(neurona_destino, peso)

    # Variables para graficar
    tiempos = []
    num_disparos_por_tiempo = []
    potenciales_membrana = [[] for _ in range(NUM_NEURONAS_A_GRAFICAR)]
    pesos_sinapticos_ejemplo = []
    neuronas_a_seguir = [corteza.neuronas[i] for i in range(NUM_NEURONAS_A_GRAFICAR)]

    # Simular actividad neuronal
    for t in range(TIEMPO_SIMULACION):
        tiempos.append(t)
        num_disparos_en_t = 0
        neuronas_disparando = []

        for estructura in estructuras:
            for neurona in estructura.neuronas:
                if random.random() < PROBABILIDAD_DISPARO:
                    if neurona.disparar(t):
                        neuronas_disparando.append(neurona)
                        num_disparos_en_t += 1
        num_disparos_por_tiempo.append(num_disparos_en_t)

        for neurona_disparando in neuronas_disparando:
            for neurona_destino, peso in neurona_disparando.conexiones_sinapticas:
                neurona_destino.recibir_estimulo(peso)
                neurona_destino.aplicar_plasticidad(t, neurona_disparando)

        for estructura in estructuras:
            for neurona in estructura.neuronas:
                if neurona.potencial_membrana > -70:
                    neurona.potencial_membrana -= DECAY_RATE
        #Datos para graficar
        for i, neurona in enumerate(neuronas_a_seguir):
            potenciales_membrana[i].append(neurona.potencial_membrana)
        if t == TIEMPO_SIMULACION-1: #solo al final de la simulación.
          for neurona_destino, peso in neuronas_a_seguir[0].conexiones_sinapticas:
              pesos_sinapticos_ejemplo.append(peso)

    # Simular EEG
    simulador_eeg = SimuladorEEG(duracion=10, tasa_muestreo=250)
    eeg = simulador_eeg.simular_eeg()

    # --- GRÁFICAS ---
    plt.figure(figsize=(12, 6))
    plt.plot(tiempos, num_disparos_por_tiempo)
    plt.title("Número de Disparos por Unidad de Tiempo")
    plt.xlabel("Tiempo (ms)")
    plt.ylabel("Número de Disparos")
    plt.show()

    plt.figure(figsize=(12, 6))
    for i, potencial in enumerate(potenciales_membrana):
        plt.plot(tiempos, potencial, label=f"Neurona {i+1}")
    plt.title("Potencial de Membrana de Múltiples Neuronas")
    plt.xlabel("Tiempo (ms)")
    plt.ylabel("Potencial de Membrana (mV)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.hist(pesos_sinapticos_ejemplo, bins = 20)
    plt.title("Histograma de pesos sinapticos de Neurona Ejemplo (Corteza[0])")
    plt.xlabel("Peso sinaptico")
    plt.ylabel("Frecuencia")
    plt.show()

    simulador_eeg.visualizar_eeg(eeg)

if __name__ == "__main__":
    main()