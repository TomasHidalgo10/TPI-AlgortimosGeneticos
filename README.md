# Trabajo Práctico Integrador - Algoritmos Genéticos

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![DEAP](https://img.shields.io/badge/DEAP-1.4%2B-green.svg)](https://deap.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/pandas-2.0%2B-darkblue.svg)](https://pandas.pydata.org/)

## Predicción de la condición de desocupación en la PEA de Argentina mediante Machine Learning con selección de variables optimizada por Algoritmos Genéticos

- **Materia:** Algoritmos Genéticos
- **Carrera:** Ingeniería en Sistemas de Información
- **Institución:** Universidad Tecnológica Nacional (UTN)

---

## 📋 1. Descripción del Proyecto

Este proyecto desarrolla un modelo predictivo basado en **Machine Learning** capaz de clasificar a una persona de la **Población Económicamente Activa (PEA)** como **ocupada** o **desocupada** utilizando características sociodemográficas provenientes de la Encuesta Permanente de Hogares (**EPH**).

Para optimizar el conjunto de características predictoras, se implementa un **Algoritmo Genético (AG)** que selecciona el subconjunto de variables óptimo, maximizando el desempeño métrico (F1-score sobre la clase minoritaria) y penalizando la redundancia/dimensionalidad.

### Comparativa Central
1. **Modelo Base:** Random Forest utilizando todas las variables disponibles tras la codificación.
2. **Modelo Optimizado:** Random Forest entrenado únicamente con el subconjunto de variables seleccionado por el Algoritmo Genético.

### Finalidad Práctica y Alcance Ético
El modelo **no** tiene como finalidad clasificar automáticamente a las personas ni tomar decisiones automáticas sobre ellas. Su propósito es identificar perfiles o grupos con mayor vulnerabilidad o probabilidad de desocupación para orientar políticas públicas de empleo:
- Capacitación laboral y formación técnica
- Programas de inserción y reorientación profesional
- Orientación activa en la búsqueda de empleo
- Seguimiento y diagnóstico preventivo

> **Nota:** La predicción debe interpretarse siempre como una herramienta de apoyo y diagnóstico. Toda decisión final corresponde a organismos e instancias humanas.

---

## 📊 2. Fuente de Datos

**INDEC** — Instituto Nacional de Estadística y Censos de la República Argentina.

- **Encuesta:** Encuesta Permanente de Hogares (EPH) - Total urbano
- **Período:** Tercer trimestre de 2025 (3T 2025)
- **Portal de Datos Abiertos:** [Bases de Datos INDEC](https://www.indec.gob.ar/indec/web/Institucional-Indec-BasesDeDatos)

Se utilizan los microdatos individuales (`personas_tot.urb_3T_2025.txt`) junto con las especificaciones del diseño de registro oficial de la EPH.

---

## 🚀 3. Estructura del Repositorio

```text
├── codigo/
│   ├── __init__.py                # Inicializador del paquete
│   ├── configuracion.py           # Parámetros globales, rutas y constantes
│   ├── carga_datos.py             # Carga y filtrado de la PEA
│   ├── preprocesamiento_datos.py  # Limpieza, imputación y codificación (OHE/Ordinal)
│   ├── modelo_base.py             # Random Forest base con todas las variables
│   ├── algoritmo_genetico.py      # Selección de variables con DEAP
│   ├── evaluacion_modelo.py       # Evaluación comparativa y métricas
│   └── visualizaciones.py         # Generación de gráficos y curvas
│
├── datos/
│   └── originales/                # Microdatos oficiales EPH (INDEC)
│
├── documentacion/
│   └── informe_final.md           # Informe técnico y metodológico detallado
│
├── notebooks/
│   └── analisis_modelo.ipynb      # Notebook Jupyter con análisis interactivo
│
├── resultados/
│   ├── graficos/                  # Figuras y curvas exportadas (.png)
│   ├── metricas/                  # Tablas de métricas (.csv, .json)
│   └── reporte_resultados.docx    # Informe ejecutivo exportado
│
├── ejecutar_proyecto.py           # Script principal de ejecución
├── generar_reporte_docx.py        # Generador del documento Word (.docx)
├── requisitos.txt                 # Dependencias del entorno Python
├── LEEME.md                       # Documentación en español
└── README.md                      # Presentación principal del repositorio
```

---

## ⚙️ 4. Instalación y Requisitos

### Requisitos Previos
- Python 3.10 o superior
- Administrador de paquetes `pip`

### Instalación de Dependencias

```bash
# Clonar el repositorio
git clone https://github.com/TomasHidalgo10/TPI-AlgortimosGeneticos.git
cd TPI-AlgortimosGeneticos

# Crear y activar entorno virtual (opcional pero recomendado)
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate

# Instalar dependencias
pip install -r requisitos.txt
```

---

## 💻 5. Modos de Ejecución

### Opción A: Flujo Completo (Consola)
Ejecuta todo el pipeline (carga, preprocesamiento, modelo base, AG, optimizado, métricas y gráficos):

```bash
python ejecutar_proyecto.py
```

### Opción B: Generación del Reporte Word (.docx)
Genera el informe formateado con tablas de métricas y gráficos embebidos:

```bash
python generar_reporte_docx.py
```

### Opción C: Notebook Interactivo
Exploración visual e interactiva paso a paso:

```bash
jupyter notebook notebooks/analisis_modelo.ipynb
```

---

## 📈 6. Resultados y Métricas Generadas

Tras la ejecución, se actualizan los artefactos en `resultados/`:

- **Gráficos (`resultados/graficos/`):**
  1. `01_distribucion_clases.png`: Distribución de la condición laboral en la PEA.
  2. `02_evolucion_fitness.png`: Evolución del fitness a lo largo de las generaciones.
  3. `03_evolucion_variables.png`: Reducción del conteo de variables seleccionadas.
  4. `04_comparacion_metricas.png`: Comparativa Modelo Base vs. Modelo Optimizado.
  5. `05_matriz_confusion_base.png`: Matriz de confusión del Modelo Base.
  6. `06_matriz_confusion_optimizado.png`: Matriz de confusión del Modelo Optimizado.
  7. `07_importancia_variables.png`: Importancia de características (Feature Importance Gini).
  8. `08_analisis_exploratorio.png`: Cruces sociodemográficos (sexo, edad, educación, estado civil).

- **Métricas (`resultados/metricas/`):**
  - `comparacion_modelos.csv`: Comparación numérica exhaustiva.
  - `variables_seleccionadas.csv`: Detalle de predictores retenidos vs descartados.
  - `resultados_completos.json`: Métricas completas para interoperabilidad.
  - `evolucion_ag.csv`: Registro paso a paso del proceso evolutivo.

---

## 🔬 7. Metodología y Reproducibilidad

- **Semilla fija:** `random_state = 42` para garantizar total reproducibilidad.
- **Estratificación:** Muestreo balanceado en particiones train/test y Stratified K-Fold CV (5 folds).
- **Aislamiento de prueba (No Data Leakage):** Los encoders y la optimización genética operan exclusivamente sobre el conjunto de entrenamiento.
- **Tratamiento de Desbalance:** Ponderación `class_weight='balanced'` y optimización orientada al F1-Score / Recall de la clase desocupada.
- **Función de Fitness con Penalización de Parsimonia:**
  $$\text{Fitness} = \text{F1}_{\text{desocupado}} - \lambda \cdot \left(\frac{N_{\text{seleccionadas}}}{N_{\text{totales}}}\right)$$

---

## 📚 8. Licencia y Uso Académico

Proyecto desarrollado con fines académicos en el marco de la materia **Algoritmos Genéticos**. Los microdatos pertenecen al **INDEC**.
