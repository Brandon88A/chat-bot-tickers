# Red neuronal para clasificar mensajes y tickets

Perceptrón multicapa (Keras/TensorFlow) que clasifica mensajes de clientes en 4 categorías: *Cita, Facturación, Soporte y Ventas*.

## Archivos

- red_neuronal_tickets.py: entrena la red, muestra la accuracy, genera la gráfica y prueba mensajes nuevos.
- A7_Experimentacion_Red_Neuronal.pdf: documento explicativo de la experimentación.

## Cómo ejecutarlo

bash
pip install tensorflow scikit-learn matplotlib
python red_neuronal_tickets.py


También funciona en Google Colab.

## Red neuronal

| Capa | Configuración |
|------|---------------|
| Entrada | Vector TF-IDF |
| Densa 1 | 64 neuronas, ReLU |
| Dropout | 0.3 |
| Densa 2 | 32 neuronas, ReLU |
| Salida | 4 neuronas, Softmax |

Optimizador Adam, pérdida sparse_categorical_crossentropy, 80% entrenamiento y 20% prueba.

## Resultados

- Accuracy en prueba: pendiente
- Pérdida en prueba: pendiente
