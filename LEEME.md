# Trabajo Práctico Integrador - Algoritmos Genéticos

## Predicción de la condición de desocupación en la PEA de Argentina mediante Machine Learning con selección de variables optimizada por Algoritmos Genéticos

**Materia:** Algoritmos Genéticos  
**Carrera:** Ingeniería en Sistemas de Información  

---

## 1. Descripción del proyecto

Este proyecto desarrolla un modelo predictivo basado en Machine Learning capaz de clasificar a una persona de la Población Económicamente Activa (PEA) como **ocupada** o **desocupada** utilizando sus características sociodemográficas.

Se utiliza un **Algoritmo Genético** para optimizar la selección de variables predictoras, buscando el subconjunto que maximice el desempeño del modelo.

El proyecto compara:
- **Modelo base:** Random Forest con todas las variables disponibles.
- **Modelo optimizado:** Random Forest con las variables seleccionadas por el Algoritmo Genético.

### Finalidad práctica

El modelo **no** tiene como finalidad clasificar automáticamente a las personas ni tomar decisiones sobre ellas. Su propósito es identificar perfiles o grupos con mayor probabilidad de desocupación para que organismos públicos puedan orientar políticas activas de empleo como:

- Capacitación laboral
- Formación profesional
- Orientación para la búsqueda de empleo
- Programas de inserción laboral
- Seguimiento preventivo

La predicción debe interpretarse como un **indicador de apoyo**. La decisión final corresponde a organismos y responsables humanos.

---

## 2. Fuente de datos

**INDEC** - Instituto Nacional de Estadística y Censos de la República Argentina

- **Encuesta:** Encuesta Permanente de Hogares (EPH) - Total urbano
- **Período:** Tercer trimestre de 2025
- **Página de descarga:** https://www.indec.gob.ar/indec/web/Institucional-Indec-BasesDeDatos

Se utilizan los microdatos individuales (`personas_tot.urb_3T_2025.txt`) y el documento de diseño de registro oficial para la interpretación de las variables.

---

## 3. Cómo obtener los datos

1. Ingresar a la [página de bases de datos del INDEC](https://www.indec.gob.ar/indec/web/Institucional-Indec-BasesDeDatos).
2. Buscar la sección **Encuesta Permanente de Hogares (EPH) total urbano**.
3. Ir a **Microdatos (2025)** → **Bases del tercer trimestre 2025** → **Formato TXT**.
4. Descargar el archivo comprimido y extraer los archivos.
5. Colocar el archivo `personas_tot.urb_3T_2025.txt` en la carpeta `datos/originales/`.

---

## 4. Estructura de carpetas

```
tpi-Ageneticos/
│
├── datos/
│   ├── originales/                    # Datos descargados del INDEC
│   │   └── personas_tot.urb_3T_2025.txt
│   └── procesados/                    # Datos procesados (generados)
│
├── notebooks/
│   └── analisis_modelo.ipynb          # Notebook con análisis completo
│
├── codigo/
│   ├── __init__.py                    # Inicializador del paquete
│   ├── configuracion.py               # Parámetros y constantes
│   ├── carga_datos.py                 # Carga y filtrado de datos
│   ├── preprocesamiento_datos.py      # Limpieza y codificación
│   ├── modelo_base.py                 # Modelo Random Forest base
│   ├── algoritmo_genetico.py          # AG para selección de variables
│   ├── evaluacion_modelo.py           # Evaluación y comparación
│   └── visualizaciones.py            # Generación de gráficos
│
├── resultados/
│   ├── graficos/                      # Gráficos generados
│   ├── metricas/                      # Métricas y tablas
│   └── modelos/                       # Modelos guardados
│
├── documentacion/
│   └── informe_final.md               # Informe completo del TP
│
├── ejecutar_proyecto.py               # Script principal de ejecución
├── generar_reporte_docx.py            # Generador de reporte Word (.docx)
├── requisitos.txt                     # Dependencias de Python
├── LEEME.md                           # Documentación en español
└── README.md                          # Presentación principal para GitHub
```


---

## 5. Instalación de dependencias

```bash
pip install -r requisitos.txt
```

Librerías utilizadas:
- **pandas**: manipulación de datos
- **numpy**: operaciones numéricas
- **scikit-learn**: modelos de Machine Learning y preprocesamiento
- **matplotlib**: visualizaciones
- **seaborn**: visualizaciones estadísticas
- **deap**: implementación del Algoritmo Genético
- **python-docx**: generación de reportes en formato Word (.docx)

---

## 6. Cómo ejecutar el proyecto

### Opción 1: Script principal

```bash
python ejecutar_proyecto.py
```

Este script ejecuta todo el flujo:
1. Carga los microdatos de la EPH
2. Filtra la PEA
3. Crea la variable objetivo
4. Limpia y preprocesa los datos
5. Entrena el modelo base
6. Ejecuta el Algoritmo Genético
7. Entrena el modelo optimizado
8. Compara ambos modelos
9. Genera visualizaciones
10. Guarda todos los resultados

### Opción 2: Generación del Reporte Word (.docx)

```bash
python generar_reporte_docx.py
```

Genera un documento Word completo (`resultados/reporte_resultados.docx`) con todo el análisis, tablas de métricas y gráficos embebidos.

### Opción 3: Notebook

```bash
jupyter notebook notebooks/analisis_modelo.ipynb
```

El notebook contiene el mismo flujo con explicaciones detalladas paso a paso.

---

## 7. Descripción de archivos

| Archivo | Descripción |
|---------|-------------|
| `configuracion.py` | Parámetros centrales: rutas, semillas, parámetros del modelo y del AG |
| `carga_datos.py` | Lectura del archivo TXT, filtrado de la PEA, creación del target |
| `preprocesamiento_datos.py` | Limpieza de valores especiales, codificación, división train/test |
| `modelo_base.py` | Random Forest con todas las variables, evaluación con múltiples métricas |
| `algoritmo_genetico.py` | Implementación con DEAP: cromosomas binarios, fitness con CV |
| `evaluacion_modelo.py` | Comparación modelo base vs. optimizado, tablas y análisis |
| `visualizaciones.py` | 8 gráficos en español |
| `ejecutar_proyecto.py` | Orquesta todo el flujo de ejecución |
| `generar_reporte_docx.py` | Generador del reporte de resultados en formato .docx |

---

## 8. Resultados generados

Tras la ejecución, se generan los siguientes archivos en `resultados/`:

### Gráficos (`resultados/graficos/`)
1. Distribución de clases (ocupados/desocupados)
2. Evolución del fitness del AG
3. Evolución de la cantidad de variables seleccionadas
4. Comparación de métricas entre modelos
5. Matriz de confusión del modelo base
6. Matriz de confusión del modelo optimizado
7. Importancia de variables
8. Análisis exploratorio

### Métricas (`resultados/metricas/`)
- `comparacion_modelos.csv`: tabla comparativa
- `variables_seleccionadas.csv`: variables del AG
- `resultados_completos.json`: métricas completas en JSON
- `evolucion_ag.csv`: evolución por generación
- `importancia_variables_base.csv`: importancia de variables
- `importancia_variables_optimizado.csv`: importancia del modelo optimizado

---

## 9. Consideraciones metodológicas

- **Semilla de reproducibilidad:** `random_state = 42` en todos los componentes.
- **División estratificada:** se preserva la proporción de clases en train/test.
- **Sin data leakage:** el conjunto de prueba no se utiliza durante la selección de variables ni la optimización.
- **Validación cruzada:** el fitness del AG se calcula con Stratified K-Fold CV (5 folds) sobre el conjunto de entrenamiento.
- **Desbalance de clases:** se utiliza `class_weight='balanced'` y se presta especial atención al Recall y F1 de la clase desocupada.
- **NIVEL_ED como ordinal:** se trata como variable ordinal porque los niveles educativos poseen un orden natural.
- **Correlación ≠ causalidad:** el modelo identifica relaciones predictivas, no causales.

---

## 10. Fuentes

- INDEC - Instituto Nacional de Estadística y Censos. Encuesta Permanente de Hogares (EPH). Microdatos del tercer trimestre de 2025. https://www.indec.gob.ar/indec/web/Institucional-Indec-BasesDeDatos
- Diseño de registro y estructura para las bases de microdatos de la EPH (3er trimestre 2025). INDEC.

---

## 11. Nota importante

Este proyecto tiene finalidad exclusivamente **académica**. Los resultados no deben utilizarse para tomar decisiones automáticas sobre personas ni para negar derechos o beneficios. El modelo es una herramienta de apoyo y la decisión final debe permanecer bajo supervisión humana.
