# AI Project - Regression Marketing

## Predicción de ingresos de campañas de marketing mediante Machine Learning

Este proyecto desarrolla un modelo de Machine Learning capaz de **predecir el ingreso esperado (Revenue) de una campaña de marketing antes de su lanzamiento**, utilizando únicamente variables disponibles durante la fase de planificación.

El objetivo es proporcionar una herramienta que ayude a equipos de Marketing y Performance a tomar decisiones de inversión antes de ejecutar una campaña.

---

# Problema de negocio

Antes de lanzar una campaña publicitaria es habitual preguntarse:

- ¿Cuánto ingreso podría generar?
- ¿Vale la pena invertir este presupuesto?
- ¿Qué configuración ofrece mejores resultados?

Este proyecto intenta responder a estas preguntas mediante un modelo predictivo entrenado con datos históricos de campañas.

---

# Objetivos

- Analizar un dataset de campañas digitales.
- Realizar un análisis exploratorio de datos (EDA).
- Construir y comparar modelos de regresión.
- Optimizar el mejor modelo.
- Evaluar su capacidad de generalización.
- Desarrollar una aplicación interactiva para realizar predicciones.

---

# Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-Learn
- Optuna
- Streamlit
- Joblib
- Matplotlib
- Git
- GitHub

---

# Estructura del proyecto

```text
.
├── app/
│   └── streamlit_app.py
│
├── data/
│
├── docs/
│
├── images/
│
├── models/
│   └── campaign_revenue_forecast_model.joblib
│
├── notebooks/
│   ├── 01_dataset_assessment.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   └── 03_modeling.ipynb
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   ├── evaluation.py
│   ├── train.py
│   └── features.py
│
├── tests/
│
├── README.md
└── requirements.txt
```

---

# Flujo del proyecto

- Dataset Assessment
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Comparación de modelos
- Optimización mediante Optuna
- Selección del modelo final
- Evaluación de métricas
- Análisis de Overfitting
- Persistencia del modelo
- Aplicación Streamlit

---

# Modelo seleccionado

El modelo final es un:

**Random Forest Regressor**

optimizado mediante búsqueda de hiperparámetros y serializado utilizando Joblib para su posterior reutilización.

---

# Resultados obtenidos

| Métrica | Resultado |
|----------|-----------|
| R² | 0.71 |
| RMSE | 1614.45 |
| MAE | 267.48 |
| Overfitting Gap | ~10 % |

Estos resultados indican que el modelo es capaz de explicar aproximadamente el **71 % de la variabilidad del Revenue** utilizando únicamente información disponible antes del lanzamiento de la campaña.

---

# Aplicación Streamlit

La aplicación permite introducir los parámetros de planificación de una campaña y obtener una predicción del Revenue esperado.

## Ejecutar la aplicación

```bash
streamlit run app/streamlit_app.py
```

Una vez iniciada, la aplicación estará disponible en:

```text
http://localhost:8501
```

Desde la interfaz es posible:

- Configurar los parámetros de una campaña.
- Introducir el presupuesto previsto.
- Obtener una estimación del Revenue esperado.
- Validar automáticamente los datos introducidos.

---

# Entrenar nuevamente el modelo

```bash
python -m src.train
```

---

# Ejecutar los tests

```bash
python -m pytest
```

---

# Estado del proyecto

## Completado

- Dataset Assessment
- Exploratory Data Analysis
- Ingeniería de características
- Comparación de modelos
- Optimización de hiperparámetros
- Selección del modelo final
- Evaluación del modelo
- Persistencia del modelo
- API de predicción
- Aplicación Streamlit (MVP)

## Próximas mejoras

- Registro de predicciones
- Dockerización
- Despliegue
- Documentación final

---

# Autora

**Gabriela Granja**

Bootcamp de Inteligencia Artificial – Factoría F5

Junio 2026