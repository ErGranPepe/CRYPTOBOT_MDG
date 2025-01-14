import tkinter as tk
import numpy as np
import tensorflow as tf
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Crear datos de entrenamiento
np.random.seed(0)
X = np.random.rand(100, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Crear modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Crear ventana
window = tk.Tk()
window.title("Red Neuronal Simple")

# Crear gráfico
fig = Figure(figsize=(6, 6))
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=window)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

# Función para actualizar el gráfico
def update_plot():
    ax.clear()
    ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Clase 0')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Clase 1')
    
    # Dibujar límite de decisión
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, levels=np.linspace(0, 1, 3))
    
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.legend()
    canvas.draw()

# Función para entrenar el modelo
def train_step():
    history = model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    accuracy = history.history['accuracy'][-1]
    loss = history.history['loss'][-1]
    accuracy_label.config(text=f"Precisión: {accuracy:.4f}")
    loss_label.config(text=f"Pérdida: {loss:.4f}")
    update_plot()
    window.after(100, train_step)

# Botón para iniciar entrenamiento
start_button = tk.Button(window, text="Iniciar Entrenamiento", command=train_step)
start_button.pack()

# Etiquetas para mostrar métricas
accuracy_label = tk.Label(window, text="Precisión: ")
accuracy_label.pack()
loss_label = tk.Label(window, text="Pérdida: ")
loss_label.pack()

update_plot()
window.mainloop()
