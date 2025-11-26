# 🧠 Neural Networks - Boston Housing Price Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow?style=for-the-badge&logo=scikit-learn)

**Sistema de predicción de precios de viviendas usando Redes Neuronales Artificiales**

[Características](#-características) • [Instalación](#-instalación) • [Uso](#-uso) • [Modelos](#-modelos) • [Resultados](#-resultados)

</div>

---

## 📋 Descripción

Este proyecto implementa diferentes arquitecturas de **Redes Neuronales Artificiales (RNA)** para resolver problemas de regresión y clasificación. El caso principal es la predicción de precios de viviendas utilizando el famoso dataset **Boston Housing**, además de incluir modelos de complejidad progresiva para el aprendizaje de funciones matemáticas.

Desarrollado como parte de la **2ª Práctica de Sistemas Inteligentes**, este proyecto demuestra:
- Implementación de redes neuronales desde cero con TensorFlow/Keras
- Técnicas de preprocesamiento y normalización de datos
- Callbacks avanzados para optimización del entrenamiento
- Evaluación de modelos con métricas estándar
- Visualización de resultados y métricas de rendimiento

## ✨ Características

### 🎯 Modelos Implementados

#### 1. **Predictor Simple** (`predictor_simple.py`)
- Red neuronal básica con 1 capa
- Aprende relaciones lineales: `y = 2x - 1`
- Ideal para entender conceptos fundamentales
- Optimizador: SGD (Descenso de Gradiente Estocástico)

#### 2. **Predictor Complejo** (`predictor_complejo.py`)
- Red profunda con 3 capas ocultas (64, 32, 16 neuronas)
- Aprende relaciones no lineales: `y = 3x² + 2x + 1`
- Activación ReLU
- Optimizador: Adam

#### 3. **Predictor Boston Housing** (`predictor.py`) ⭐
- Predicción de precios de viviendas
- Arquitectura: 64 → 32 → 16 → 1
- Normalización con StandardScaler
- División train/test (80/20)
- Métricas: MSE, R²
- Visualizaciones completas

#### 4. **MNIST con Callbacks** (`entrenamiento_con_callbacks.py`)
- Clasificación de dígitos escritos a mano
- Callbacks avanzados:
  - **EarlyStopping**: Detiene el entrenamiento cuando no hay mejora
  - **ModelCheckpoint**: Guarda el mejor modelo
  - **ReduceLROnPlateau**: Ajusta la tasa de aprendizaje dinámicamente
- Arquitectura: 128 → Dropout → 64 → 10 (Softmax)

### 📊 Funcionalidades

- ✅ Preprocesamiento automático de datos
- ✅ Normalización de características
- ✅ División train/validation/test
- ✅ Entrenamiento con callbacks inteligentes
- ✅ Métricas de evaluación (MSE, R², Accuracy)
- ✅ Gráficas de pérdida y precisión
- ✅ Comparación valores reales vs predicciones
- ✅ Guardado de mejores modelos

## 🚀 Instalación

### Requisitos Previos
- **Python 3.11** o superior
- **pip** (gestor de paquetes de Python)

### Dependencias
```bash
# Crear entorno virtual (recomendado)
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install tensorflow
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
```

### Clonar el Repositorio
```bash
git clone https://github.com/alokenveo/neural-networks-housing.git
cd neural-networks-housing
```

## 💻 Uso

### Ejecutar el Predictor Principal (Boston Housing)
```bash
python main.py
```

O directamente:
```bash
python scripts/predictor.py
```

### Ejecutar Modelos Individuales
```bash
# Modelo simple (relación lineal)
python scripts/predictor_simple.py

# Modelo complejo (relación cuadrática)
python scripts/predictor_complejo.py

# Modelo MNIST con callbacks
python scripts/entrenamiento_con_callbacks.py
```

### Salida Esperada
```
Cargando el dataset Boston Housing...
Evaluando el modelo...

Pérdida en el conjunto de prueba: 21.3456

Comparación de los valores reales con las predicciones:
Real: 24.00, Predicho: 22.45
Real: 21.60, Predicho: 20.87
Real: 34.70, Predicho: 33.12
...

Error cuadrático medio (MSE): 21.3456
Coeficiente de determinación (R²): 0.7234
```

## 🏛️ Arquitectura del Proyecto
```
neural-networks-housing/
├── .idea/                      # Configuración de PyCharm
│   ├── inspectionProfiles/
│   ├── .gitignore
│   ├── misc.xml
│   ├── modules.xml
│   └── vcs.xml
├── scripts/
│   ├── predictor.py           # Predictor principal (Boston Housing) ⭐
│   ├── predictor_simple.py    # Modelo básico (y = 2x - 1)
│   ├── predictor_complejo.py  # Modelo avanzado (y = 3x² + 2x + 1)
│   └── entrenamiento_con_callbacks.py  # MNIST con callbacks
├── main.py                    # Punto de entrada principal
├── README.md                  # Este archivo
└── best_model_mnist.keras    # Mejor modelo guardado (generado)
```

## 🧮 Modelos en Detalle

### Predictor Boston Housing

#### Arquitectura
```python
model = tf.keras.Sequential([
    Dense(64, activation='relu', input_shape=(13,)),  # Capa entrada
    Dense(32, activation='relu'),                     # Capa oculta 1
    Dense(16, activation='relu'),                     # Capa oculta 2
    Dense(1)                                          # Capa salida
])
```

#### Dataset
- **Características**: 13 variables (criminalidad, zonas residenciales, edad de viviendas, etc.)
- **Target**: Precio medio de viviendas en miles de dólares
- **Instancias**: 506 viviendas en Boston
- **Fuente**: Carnegie Mellon University Statistics

#### Métricas
- **MSE (Mean Squared Error)**: Mide el error promedio al cuadrado
- **R² (Coeficiente de determinación)**: Indica qué tan bien el modelo explica la variabilidad (0-1)

### Modelo MNIST

#### Arquitectura
```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.5),              # Previene overfitting
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 clases (dígitos 0-9)
])
```

#### Callbacks Utilizados
- **EarlyStopping**: `patience=5` - Detiene tras 5 épocas sin mejora
- **ModelCheckpoint**: Guarda automáticamente el mejor modelo
- **ReduceLROnPlateau**: Reduce learning rate cuando se estanca

## 📈 Resultados Esperados

### Boston Housing Predictor
- **MSE**: ~20-25 (dependiendo de la semilla aleatoria)
- **R²**: ~0.70-0.80
- **Tiempo de entrenamiento**: 10-30 segundos (100 épocas)

### Predictor Simple
- Aprende perfectamente la relación lineal `y = 2x - 1`
- **Predicciones**:
  - x=5 → y≈9
  - x=10 → y≈19
  - x=15 → y≈29

### Predictor Complejo
- Aprende la relación cuadrática `y = 3x² + 2x + 1`
- **R²**: ~0.99 (ajuste casi perfecto)
- **Tiempo**: ~5-10 segundos (1000 épocas)

### MNIST
- **Accuracy**: ~97-98% en validación
- **Épocas**: Generalmente converge en 10-15 épocas con early stopping

## 📊 Visualizaciones

El proyecto genera automáticamente:

1. **Gráfica de Pérdida**: Evolución del loss en entrenamiento y validación
2. **Gráfica de Precisión**: Para modelos de clasificación (MNIST)
3. **Scatter Plot**: Comparación valores reales vs predicciones
4. **Comparación Numérica**: Tabla con primeras 10 predicciones

## 🛠️ Tecnologías Utilizadas

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| Python | 3.11 | Lenguaje principal |
| TensorFlow | 2.x | Framework de Deep Learning |
| Keras | Incluido en TF | API de alto nivel para redes neuronales |
| NumPy | Latest | Operaciones numéricas |
| Pandas | Latest | Manipulación de datos |
| scikit-learn | Latest | Preprocesamiento y métricas |
| Matplotlib | Latest | Visualizaciones |

## 🎓 Conceptos Aprendidos

- ✅ Arquitecturas de redes neuronales
- ✅ Funciones de activación (ReLU, Softmax)
- ✅ Optimizadores (SGD, Adam)
- ✅ Funciones de pérdida (MSE, Categorical Crossentropy)
- ✅ Regularización (Dropout)
- ✅ Callbacks y control de entrenamiento
- ✅ Normalización de datos
- ✅ Métricas de evaluación
- ✅ Overfitting y underfitting
- ✅ Validación cruzada

## 🚧 Próximas Mejoras

- [ ] Implementar Grid Search para hiperparámetros
- [ ] Añadir más datasets (California Housing, Wine Quality)
- [ ] Crear interfaz gráfica con Streamlit
- [ ] Implementar redes convolucionales (CNN)
- [ ] Añadir redes recurrentes (LSTM) para series temporales
- [ ] Exportar modelos a formatos de producción (ONNX, TFLite)
- [ ] Dashboard interactivo con Plotly
- [ ] API REST con FastAPI para predicciones

## 📚 Referencias

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Guide](https://keras.io/guides/)
- [Boston Housing Dataset](http://lib.stat.cmu.edu/datasets/boston)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)

## 👨‍💻 Autor

**Alfredo Mituy Okenve**  
*2ª Práctica de Sistemas Inteligentes*

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la Licencia MIT.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para cambios importantes:

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/mejora`)
3. Commit cambios (`git commit -m 'Añadir mejora'`)
4. Push a la rama (`git push origin feature/mejora`)
5. Abre un Pull Request

## 🐛 Problemas Conocidos

- Los warnings de TensorFlow están suprimidos para mejor legibilidad
- El dataset Boston Housing está deprecated en scikit-learn (se carga desde URL)
- Los modelos no se persisten por defecto (excepto MNIST con callbacks)

## 💡 Consejos de Uso

- **Para experimentar**: Modifica las arquitecturas en los archivos `predictor_*.py`
- **Para aprender**: Empieza con `predictor_simple.py` y avanza progresivamente
- **Para producción**: Usa el modelo con callbacks y guarda el mejor resultado
- **Para debugging**: Activa `verbose=1` en `model.fit()` para ver el progreso

---

<div align="center">

**¿Preguntas o sugerencias?**  
Abre un [issue](https://github.com/alokenveo/neural-networks-housing/issues) en GitHub

Hecho con 🧠 y mucho ☕

</div>
