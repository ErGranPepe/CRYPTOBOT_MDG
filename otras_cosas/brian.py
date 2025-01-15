from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

# Configuración del entorno
start_scope()
defaultclock.dt = 0.1*ms  # Paso temporal pequeño para mejorar la precisión

# Parámetros generales
N_excitatory = 8000  # Neuronas excitatorias
N_inhibitory = 2000  # Neuronas inhibitorias
N = N_excitatory + N_inhibitory  # Total de neuronas

# Parámetros de la neurona (usando características más realistas)
El = -60*mV  # Potencial de reposo
Vr = -65*mV  # Potencial de reinicio
Vt = -50*mV  # Umbral de activación
taum = 20*ms  # Constante de tiempo de membrana
C = 100*pF  # Capacitancia de la membrana
gL = 10*nS  # Conductancia de fuga
Ee = 0*mV  # Potencial de excitación
Ei = -80*mV  # Potencial de inhibición

# Parámetros de sinapsis
taue = 5*ms  # Constante de tiempo para sinapsis excitatorias
taui = 10*ms  # Constante de tiempo para sinapsis inhibitorias

# Precalcular valores aleatorios para evitar cálculos repetidos
rand_a = (0.5 + np.random.rand(N) * 0.5) * nS
rand_b = (0.5 + np.random.rand(N) * 0.1) * nA
rand_tauw = (150 + np.random.rand(N) * 150) * ms
rand_v = El + (Vt-El)*np.random.rand(N)
rand_ge = 0.5 * nS * np.random.randn(N)  # Excitación aleatoria
rand_gi = 2 * nS * np.random.randn(N)  # Inhibición aleatoria

# Ecuaciones del modelo neuronal
eqs = '''
dv/dt = (gL*(El-v) + ge*(Ee-v) + gi*(Ei-v) + I_noise)/C : volt
dge/dt = -ge/taue : siemens
dgi/dt = -gi/taui : siemens
dw/dt = (a*(v-El) - w)/tauw : amp
I_noise : amp
a : siemens
b : amp
tauw : second
'''

# Crear el grupo de neuronas
G = NeuronGroup(N, eqs, threshold='v>Vt', reset='v=Vr; w+=b', method='exponential_euler')

# Inicializar las variables de las neuronas
G.v = rand_v
G.ge = rand_ge
G.gi = rand_gi
G.I_noise = '(randn() * 20 + 100) * pA'  # Ruido en cada paso

G.a = rand_a
G.b = rand_b
G.tauw = rand_tauw

# Sinapsis: Conexiones excitatorias e inhibitorias con neurotransmisores diferentes
Se = Synapses(G[:N_excitatory], G, 'w_syn : siemens', on_pre='ge_post += w_syn')  # Excitación
Se.connect(p=0.05)

Si = Synapses(G[N_excitatory:], G, 'w_syn : siemens', on_pre='gi_post += w_syn')  # Inhibición
Si.connect(p=0.05)

# Parámetros de plasticidad sináptica (STDP con metaplasticidad)
taupre = taupost = 20*ms
Apre = 0.01
Apost = -Apre * 1.05 * nS  # Plasticidad inversa
wmax = 1*nS  # Peso máximo sináptico

stdp = Synapses(G[:N_excitatory], G[:N_excitatory],
                '''
                w_stdp : siemens
                dapre/dt = -apre/taupre : 1 (event-driven)
                dapost/dt = -apost/taupost : 1 (event-driven)
                ''',
                on_pre='''
                ge_post += w_stdp
                apre += Apre
                w_stdp = clip(w_stdp + apost, 0*nS, wmax)
                ''',
                on_post='''
                apost += Apost
                w_stdp = clip(w_stdp + apre, 0*nS, wmax)
                ''')

stdp.connect(p=0.02)

# Monitoreo de tasa de disparo (muy importante para ver la actividad neuronal)
rate_mon = PopulationRateMonitor(G)

# Simulación del modelo
run(2000*ms, report='text')

# Graficar la tasa de disparo de la población
plt.plot(rate_mon.t/ms, rate_mon.rate/Hz)
plt.xlabel('Tiempo (ms)')
plt.ylabel('Tasa de disparo (Hz)')
plt.title('Tasa de Disparo de la Población Neuronal')
plt.show()

# Análisis adicional
print(f"Tasa de disparo promedio: {np.mean(rate_mon.rate)}")  # Usando numpy es más rápido
