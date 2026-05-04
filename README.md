📊 Sales Machine Learning Pipeline
Proyecto de análisis y predicción de ventas diseñado para simular un escenario real de trabajo en entornos corporativos.
Este pipeline automatiza el proceso completo de:

análisis de datos comerciales,
generación de indicadores clave (KPIs),
entrenamiento de modelos predictivos,
permitiendo estimar ventas futuras de manera reproducible y configurable.

💼 ¿Qué problema de negocio resuelve?
En muchas organizaciones, los análisis de ventas:

se hacen de forma manual,
dependen de notebooks difíciles de reutilizar,
no son fáciles de mantener ni repetir.

Este proyecto propone una solución automatizada que:

analiza datos históricos de ventas,
genera métricas clave para toma de decisiones,
entrena modelos de Machine Learning para predicción,
puede ejecutarse con un solo comando.


🧠 ¿Qué hace el pipeline?
De forma resumida, el sistema:

Carga y limpia datos históricos de ventas
Genera KPIs como:

ventas totales,
rentabilidad,
tiempos de envío


Crea variables relevantes para el modelo
Entrena modelos predictivos
Evalúa su desempeño
Genera resultados y logs automáticamente

Todo el proceso es reproducible y configurable, sin modificar el código.

🤖 Modelos predictivos utilizados
El pipeline permite entrenar distintos modelos, seleccionados según su idoneidad para datos de negocio:

LightGBM (modelo principal, recomendado)
XGBoost (alternativo)
Red neuronal (MLP) como opción experimental

En escenarios reales de datos tabulares, los modelos de boosting suelen ofrecer mejor desempeño y estabilidad.


Ejecución del proyecto
El pipeline se ejecuta desde la línea de comandos:

python pipeline.py --config config.yaml

Esto permite:

cambiar parámetros sin tocar el código,
reutilizar el pipeline con nuevos datos,
integrarlo fácilmente en otros flujos de trabajo.


🛠️ Tecnologías utilizadas

Python
Pandas, NumPy
Scikit‑learn
LightGBM, XGBoost
YAML (configuración)
Logging
Command Line Interface (CLI)


📝 Nota sobre entornos corporativos
Algunas funcionalidades (como redes neuronales mediante TensorFlow o Docker) están presentes como opciones, pero el pipeline fue diseñado para funcionar correctamente en entornos corporativos con restricciones técnicas, priorizando modelos y herramientas ampliamente compatibles.
