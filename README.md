# 📊 Sales Machine Learning Pipeline

Pipeline de Machine Learning end-to-end para la predicción de ventas a partir de datos históricos de negocio.

El proyecto implementa un flujo completo que incluye:
- carga y limpieza de datos
- ingeniería de características
- cálculo de KPIs
- entrenamiento y evaluación de modelos
- configuración externa mediante YAML
- ejecución vía línea de comandos (CLI)
- logging profesional

---

## 🎯 Objetivo

Construir un pipeline reproducible, configurable y orientado a entornos corporativos reales, evitando hardcodes y separando claramente la lógica de la configuración.

---

## 🧠 Modelos soportados

- LightGBM (modelo principal)
- XGBoost (alternativo)
- Red neuronal (MLP) como opción experimental

En entornos corporativos se recomienda el uso de modelos de boosting para datos tabulares.

---

## 🚀 Ejecución

```bash
python pipeline.py --config config.yaml

⚙️ Tecnologías

Python
Pandas, NumPy
Scikit-learn
LightGBM, XGBoost
YAML
Logging
CLI
