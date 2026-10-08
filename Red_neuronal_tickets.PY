clasificador de mensajes/tickets con red neuronal.
Ejecutar: pip install tensorflow scikit-learn matplotlib  ->  python red_neuronal_tickets.py
(También funciona pegado en Google Colab.)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow import keras

# 1. DATOS (agreguen más ejemplos de su proyecto: mínimo 20 por categoría)
datos = {
    "Cita": ["quiero agendar una cita", "necesito reservar un horario",
             "puedo programar una visita mañana", "quiero cambiar mi cita",
             "tienen espacio el viernes", "agéndame para la próxima semana",
             "quiero cancelar mi cita", "hay lugar para hoy en la tarde",
             "necesito una cita con el técnico", "apártame un horario el lunes",
             "quiero reprogramar mi visita", "a qué hora tienen citas disponibles"],
    "Facturación": ["no me llegó mi factura", "me cobraron de más",
                    "necesito mi comprobante de pago", "error en mi cobro",
                    "quiero una factura con RFC", "el cargo salió duplicado",
                    "me cobraron dos veces", "necesito corregir mis datos fiscales",
                    "mi pago no aparece", "envíenme el recibo de mi compra",
                    "la factura viene con otro nombre", "quiero un reembolso del cobro"],
    "Soporte": ["la app se cierra sola", "no puedo iniciar sesión",
                "el sistema marca error", "la página no carga",
                "olvidé mi contraseña", "no me deja subir archivos",
                "se queda congelada la pantalla", "no me llega el código de acceso",
                "la aplicación va muy lenta", "me sale un mensaje de falla",
                "no puedo descargar el reporte", "el sistema no guarda mis cambios"],
    "Ventas": ["quiero cotizar el servicio", "cuánto cuesta el plan",
               "tienen promociones", "quiero contratar",
               "me interesa comprar", "mándenme precios",
               "qué planes manejan", "cuánto sale por mes",
               "quiero información del paquete completo", "hacen descuento por volumen",
               "me pueden dar una cotización", "cuál es el precio del servicio anual"],
}
textos, etiquetas = [], []
nombres = list(datos.keys())
for i, (cat, frases) in enumerate(datos.items()):
    textos += frases
    etiquetas += [i] * len(frases)

# 2. PREPARACIÓN
vec = TfidfVectorizer()
X = vec.fit_transform(textos).toarray()
y = np.array(etiquetas)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. RED NEURONAL
modelo = keras.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(len(nombres), activation="softmax"),
])
modelo.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])
modelo.summary()

# 4. ENTRENAMIENTO
h = modelo.fit(X_tr, y_tr, epochs=60, validation_data=(X_te, y_te), verbose=0)

# 5. RESULTADOS
loss, acc = modelo.evaluate(X_te, y_te, verbose=0)
print(f"\nAccuracy en prueba: {acc:.2%}  |  Pérdida: {loss:.4f}")
plt.plot(h.history["accuracy"], label="entrenamiento")
plt.plot(h.history["val_accuracy"], label="prueba")
plt.xlabel("Época"); plt.ylabel("Accuracy"); plt.legend()
plt.title("Accuracy por época")
plt.savefig("grafica_accuracy.png", dpi=150)

# 6. PRUEBA CON MENSAJES NUEVOS (que no están en el entrenamiento)
print("\nPruebas con mensajes nuevos:")
for t in ["quiero apartar un lugar el lunes", "me llegó mal el cobro",
          "no abre la aplicación", "cuánto sale el servicio"]:
    p = modelo.predict(vec.transform([t]).toarray(), verbose=0)
    print(f"'{t}' -> {nombres[p.argmax()]} ({p.max():.0%})")
