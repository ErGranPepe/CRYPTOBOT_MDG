import time
import cv2
import random
import concurrent.futures
import logging
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Tuple, Union
import matplotlib.animation as animation
import threading

# Configuración del registro
logging.basicConfig(level=logging.INFO)

# ------------------------------------ CLASES BÁSICAS -------------------------------------
class Neurotransmisor:
    def __init__(self, nombre: str, efecto: str):
        self.nombre = nombre
        self.efecto = efecto  # "excitador" o "inhibidor"

class Neurona:
    def __init__(self, tipo: str, forma: str, neurotransmisor: str,
                 potencial_membrana: float = -70, umbral_activacion: float = -55):
        self.tipo = tipo
        self.forma = forma
        self.neurotransmisor = neurotransmisor
        self.potencial_membrana = potencial_membrana
        self.umbral_activacion = umbral_activacion
        self.plasticidad = 1.0
        self.conexiones_sinapticas: List[Tuple['Neurona', float]] = []
        self.historial_disparos: List[float] = []

    def disparar(self, tiempo_actual: float) -> bool:
        if self.potencial_membrana >= self.umbral_activacion:
            self.potencial_membrana = -70
            self.historial_disparos.append(tiempo_actual)
            logging.info(f"Neurona {self.tipo} dispara en t={tiempo_actual:.2f}.")
            self.propagar_senal(tiempo_actual)
            return True
        return False

    def conectar(self, neurona_destino: 'Neurona', peso: float):
        self.conexiones_sinapticas.append((neurona_destino, peso))
        logging.info(f"Conectando {self.tipo} con {neurona_destino.tipo}, peso: {peso:.2f}")

    def propagar_senal(self, tiempo_actual: float):
        influencias = np.array([peso * self.plasticidad for _, peso in self.conexiones_sinapticas])
        for (destino, _), influencia in zip(self.conexiones_sinapticas, influencias):
            destino.potencial_membrana += influencia
            logging.info(f"Señal propagada de {self.tipo} a {destino.tipo}, influencia: {influencia:.2f}")

class NeuronaPiramidal(Neurona):
    def __init__(self, neurotransmisor: str = "glutamato"):
        super().__init__(tipo="piramidal", forma="piramidal", neurotransmisor=neurotransmisor)
        self.num_dendritas = 10

class CelulaPurkinje(Neurona):
    def __init__(self, neurotransmisor: str = "GABA"):
        super().__init__(tipo="Purkinje", forma="multipolar", neurotransmisor=neurotransmisor)
        self.num_dendritas = 20

class NeuronaGranular(Neurona):
    def __init__(self, neurotransmisor: str = "glutamato"):
        super().__init__(tipo="granular", forma="esférica", neurotransmisor=neurotransmisor)
        self.num_dendritas = 4

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

    # Modificar la probabilidad de disparo
    def actividad_global(self, tiempo_actual: float):
        actividad_actual = 0
        for neurona in self.neuronas:
            if random.random() < 0.5:  # Aumentando la probabilidad a 50%
                if neurona.disparar(tiempo_actual):
                    actividad_actual += 1
        logging.info(f"Actividad en {self.nombre}: {actividad_actual} neuronas activas.")


class Corteza(EstructuraCerebral):
    def __init__(self, num_neuronas: int):
        super().__init__("Corteza", [NeuronaPiramidal() for _ in range(num_neuronas)])

class Cerebelo(EstructuraCerebral):
    def __init__(self, num_purkinje: int, num_granular: int):
        neuronas = [CelulaPurkinje() for _ in range(num_purkinje)] + [NeuronaGranular() for _ in range(num_granular)]
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

# ------------------------------------ SIMULACIÓN DE FUNCIONES CEREBRALES -------------------------------------

class AprendizajePorRefuerzo:
    def __init__(self, recompensa_max: float = 1.0):
        self.recompensa_max = recompensa_max
        self.historial: List[float] = []

    def recibir_recompensa(self, recompensa: float):
        self.historial.append(recompensa)
        if len(self.historial) > 10:
            self.historial.pop(0)
        logging.info(f"Historial de recompensas: {self.historial}")

class MecanismoAtencion:
    def __init__(self):
        self.estado_atencion = False

    def activar_atencion(self, estado: bool):
        self.estado_atencion = estado
        logging.info(f"Atención {'activada' if estado else 'desactivada'}.")

class ProcesadorSensorial:
    def __init__(self, tipo_sensor: str):
        self.tipo_sensor = tipo_sensor

    def procesar_info(self, input_sensorial: str):
        logging.info(f"Procesando {self.tipo_sensor}: {input_sensorial}")

class MemoriaTrabajo:
    def __init__(self, capacidad: int = 7):
        self.capacidad = capacidad
        self.contenido: List[str] = []

    def agregar_item(self, item: str):
        if len(self.contenido) < self.capacidad:
            self.contenido.append(item)
        else:
            self.contenido.pop(0)
            self.contenido.append(item)
        logging.info(f"Memoria de trabajo actualizada: {self.contenido}")

class TomaDecisiones:
    def __init__(self):
        self.opciones: Dict[str, float] = {}

    def evaluar_opcion(self, opcion: str, valor: float):
        self.opciones[opcion] = valor

    def tomar_decision(self) -> str:
        if self.opciones:
            decision = max(self.opciones, key=self.opciones.get)
            logging.info(f"Decisión tomada: {decision}")
            return decision
        return "No hay opciones disponibles"

class RegulacionEmocional:
    def __init__(self):
        self.estado_emocional: str = "neutral"
        self.intensidad: float = 0.5

    def cambiar_estado_emocional(self, nuevo_estado: str, nueva_intensidad: float):
        self.estado_emocional = nuevo_estado
        self.intensidad = nueva_intensidad
        logging.info(f"Estado emocional actualizado: {self.estado_emocional} (intensidad: {self.intensidad:.2f})")

# ------------------------------------ SISTEMA VISUAL -------------------------------------

class SistemaVisual:
    def __init__(self):
        pass

    def procesar_camara(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Error: No se pudo acceder a la cámara.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Error al capturar el frame.")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Sistema Visual - Entrada', gray_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

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
            "theta": OscilacionCerebral("theta", 5),
            "alpha": OscilacionCerebral("alpha", 10),
            "beta": OscilacionCerebral("beta", 20),
            "gamma": OscilacionCerebral("gamma", 40),
        }

    def simular(self):
        datos = {}
        for nombre, onda in self.ondas.items():
            datos[nombre] = onda.generar_onda(self.duracion, self.tasa_muestreo)
        return datos

    def graficar(self, datos):
        plt.figure(figsize=(10, 6))
        for nombre, valores in datos.items():
            plt.plot(valores[:500], label=f"{nombre} ({self.ondas[nombre].frecuencia} Hz)")
        plt.legend()
        plt.title("Simulación de Ondas Cerebrales")
        plt.xlabel("Tiempo (ms)")
        plt.ylabel("Amplitud")
        plt.show()

# ------------------------------------ SETUP Y PRUEBAS -------------------------------------

def setup_cerebro():
    # Crear estructuras cerebrales
    corteza = Corteza(num_neuronas=100)
    cerebelo = Cerebelo(num_purkinje=50, num_granular=100)
    hipocampo = Hipocampo(num_neuronas=80)
    ganglios_basales = GangliosBasales(num_neuronas=60)
    amigdala = Amigdala(num_neuronas=40)
    tronco_encefalico = TroncoEncefalico(num_neuronas=30)

    estructuras = [corteza, cerebelo, hipocampo, ganglios_basales, amigdala, tronco_encefalico]

    # Conexiones entre neuronas en cada estructura
    for estructura in estructuras:
        for i, neurona in enumerate(estructura.neuronas):

            if i + 1 < len(estructura.neuronas):
                neurona.conectar(estructura.neuronas[i + 1], peso=random.uniform(0.5, 1.0))

    # Simular actividad cerebral
    tiempo_simulacion = 1.0  # segundos
    for t in np.linspace(0, tiempo_simulacion, 10):
        for estructura in estructuras:
            estructura.actividad_global(t)

    # Simular EEG
    simulador_eeg = SimuladorEEG(duracion=2.0, tasa_muestreo=256)
    datos_eeg = simulador_eeg.simular()
   
if __name__ == "__main__":
    setup_cerebro()


