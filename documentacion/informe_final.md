# Predicción de la condición de desocupación en la PEA de Argentina mediante Machine Learning con selección de variables optimizada por Algoritmos Genéticos

**Trabajo Práctico Integrador**  
**Materia:** Algoritmos Genéticos  
**Carrera:** Ingeniería en Sistemas de Información  

---

## 1. Introducción

El presente trabajo aborda el desarrollo de un modelo predictivo basado en Machine Learning para clasificar personas de la Población Económicamente Activa (PEA) de Argentina como ocupadas o desocupadas, utilizando sus características sociodemográficas provenientes de la Encuesta Permanente de Hogares (EPH) del INDEC.

La particularidad del enfoque radica en la utilización de un **Algoritmo Genético (AG)** como mecanismo de **selección de variables**, con el objetivo de identificar el subconjunto óptimo de variables predictoras que maximice el desempeño del modelo, reduciendo al mismo tiempo la dimensionalidad del problema.

---

## 2. Situación problemática

El desempleo es un fenómeno multidimensional que afecta a la sociedad argentina y que depende de numerosos factores sociodemográficos como la edad, el sexo, el nivel educativo, el estado civil, la ubicación geográfica, entre otros.

Analizar estas variables de forma aislada resulta insuficiente para comprender los patrones que determinan la condición laboral de una persona. Al mismo tiempo, utilizar todas las variables disponibles en un modelo predictivo puede introducir ruido, redundancia o complejidad innecesaria, afectando negativamente el rendimiento.

---

## 3. Problema

¿Es posible predecir la condición de desocupación de una persona perteneciente a la PEA a partir de sus características sociodemográficas? ¿Puede un Algoritmo Genético ayudar a encontrar un subconjunto de variables que mejore o mantenga el desempeño predictivo del modelo, eliminando variables irrelevantes o redundantes?

---

## 4. Justificación

La identificación temprana de perfiles con mayor riesgo de desocupación tiene un valor práctico significativo para la formulación de políticas públicas. Un modelo capaz de detectar estos perfiles podría servir como herramienta de apoyo para:

- Focalizar programas de capacitación laboral
- Orientar la formación profesional
- Dirigir recursos hacia programas de inserción laboral
- Implementar seguimiento preventivo en poblaciones vulnerables

La utilización de Algoritmos Genéticos para la selección de variables aporta un componente de optimización que puede reducir la complejidad del modelo sin sacrificar su capacidad predictiva, lo cual constituye un aporte tanto metodológico como práctico.

---

## 5. Objetivo general

Construir un modelo predictivo basado en Machine Learning capaz de clasificar a una persona de la PEA como ocupada (0) o desocupada (1), utilizando características sociodemográficas de los microdatos de la EPH, y evaluar si un Algoritmo Genético puede optimizar la selección de variables predictoras.

---

## 6. Objetivos específicos

1. Cargar, explorar y comprender los microdatos de la EPH del tercer trimestre de 2025.
2. Filtrar la Población Económicamente Activa y construir la variable objetivo binaria.
3. Limpiar y preparar las variables predictoras según la documentación oficial del INDEC.
4. Entrenar un modelo base de Random Forest con todas las variables disponibles.
5. Implementar un Algoritmo Genético para la selección de variables predictoras.
6. Entrenar un modelo optimizado con las variables seleccionadas por el AG.
7. Comparar ambos modelos utilizando métricas de clasificación con especial atención a la clase desocupada.
8. Interpretar los resultados y formular conclusiones basadas en la evidencia obtenida.

---

## 7. Fuente de datos

**Fuente:** INDEC - Instituto Nacional de Estadística y Censos de la República Argentina.

**Encuesta:** Encuesta Permanente de Hogares (EPH) - Total urbano.

**Período:** Tercer trimestre de 2025.

**Archivo utilizado:** `personas_tot.urb_3T_2025.txt` (microdatos individuales).

**Página de descarga:** https://www.indec.gob.ar/indec/web/Institucional-Indec-BasesDeDatos

Los datos fueron utilizados tal como los publica el INDEC, sin modificaciones artificiales.

---

## 8. Descripción de los datos

El archivo de microdatos individuales contiene **71.322 registros** con **204 variables** correspondientes a personas encuestadas en 54 aglomerados urbanos de todo el país.

Cada registro representa una persona encuestada e incluye información sobre:
- Características demográficas (sexo, edad, estado civil)
- Nivel educativo
- Cobertura de salud
- Relación de parentesco en el hogar
- Condición de actividad (ocupado, desocupado, inactivo)
- Ubicación geográfica (aglomerado, provincia)

---

## 9. Población Económicamente Activa

La variable `ESTADO` del INDEC clasifica la condición de actividad:

| Código | Significado | Acción |
|--------|-------------|--------|
| 0 | Entrevista individual no realizada | Eliminar |
| 1 | Ocupado | Conservar |
| 2 | Desocupado | Conservar |
| 3 | Inactivo | Eliminar |
| 4 | Menor de 10 años | Eliminar |

Tras el filtrado se obtuvieron **33.384 registros** correspondientes a la PEA:
- **Ocupados:** 31.466 (94,25%)
- **Desocupados:** 1.918 (5,75%)

El ratio de desbalance es de aproximadamente **16:1**, lo cual es esperable en la realidad del mercado laboral argentino.

---

## 10. Definición de la variable objetivo

Se creó la variable binaria `desocupado`:

| ESTADO original | Variable `desocupado` | Interpretación |
|----------------|----------------------|----------------|
| 1 (Ocupado) | 0 | Clase mayoritaria |
| 2 (Desocupado) | 1 | Clase minoritaria (de interés) |

---

## 11. Preprocesamiento

### Variables predictoras seleccionadas

Se seleccionaron las siguientes variables sociodemográficas como predictoras candidatas:

| Variable | Descripción | Tipo | Tratamiento |
|----------|-------------|------|-------------|
| CH04 | Sexo | Categórica | One-Hot Encoding |
| CH06 | Edad (años cumplidos) | Numérica | Sin transformación |
| CH07 | Estado civil | Categórica | One-Hot Encoding |
| CH08 | Cobertura médica | Categórica | One-Hot Encoding |
| CH03 | Relación de parentesco | Categórica | One-Hot Encoding |
| NIVEL_ED | Nivel educativo | Ordinal | Codificación ordinal |
| CH09 | Asistencia escolar | Categórica | One-Hot Encoding |
| CH15 | Lugar de nacimiento | Categórica | One-Hot Encoding |
| AGLOMERADO | Aglomerado urbano | Categórica | One-Hot Encoding |

### Tratamiento de NIVEL_ED

Se decidió tratar `NIVEL_ED` como **variable ordinal** (no categórica) porque los niveles educativos poseen una jerarquía natural que representa mayor formación:

| Código INDEC | Nivel | Valor ordinal |
|-------------|-------|---------------|
| 7 | Sin instrucción | 0 |
| 1 | Primaria incompleta | 1 |
| 2 | Primaria completa | 2 |
| 3 | Secundaria incompleta | 3 |
| 4 | Secundaria completa | 4 |
| 5 | Superior universitaria incompleta | 5 |
| 6 | Superior universitaria completa | 6 |

Tratar esta variable como categórica pura descartaría la información de orden, lo cual no es adecuado metodológicamente.

### Limpieza de valores especiales

- **CH06 (Edad):** El código -1 indica menores de 1 año según la documentación del INDEC. En la PEA no se encontraron valores negativos, lo cual es correcto ya que la PEA comprende personas de 10 años o más.
- **CH07 (Estado civil):** Se encontraron 5 registros con código 9 (Ns/Nr), representando menos del 0,02% de los datos. Se eliminaron por no aportar información.
- **CH08 (Cobertura médica):** Se encontraron registros con código 9 (Ns/Nr). Se eliminaron.
- **CH15 (Lugar de nacimiento):** Se encontraron registros con código 9 (Ns/Nr). Se eliminaron.

### Codificación

Se utilizó `ColumnTransformer` de scikit-learn para aplicar One-Hot Encoding a las variables categóricas. El codificador se ajustó **exclusivamente con los datos de entrenamiento** para evitar data leakage.

### División de datos

- **80% entrenamiento** / **20% prueba**
- División **estratificada** (`stratify=y`) para preservar la proporción de clases
- Semilla: `random_state=42`

---

## 12. Modelo de Machine Learning

Se utilizó **Random Forest** como modelo principal por las siguientes razones:

- Maneja bien variables de distinto tipo
- Es robusto ante outliers
- No requiere normalización de variables
- Proporciona importancia de variables
- Permite compensar desbalance de clases mediante `class_weight='balanced'`

### Parámetros del modelo

| Parámetro | Valor |
|-----------|-------|
| n_estimators | 200 |
| max_depth | 15 |
| min_samples_split | 5 |
| min_samples_leaf | 2 |
| class_weight | balanced |
| random_state | 42 |

---

## 13. Algoritmo Genético

### Objetivo

Seleccionar el subconjunto de variables predictoras que maximice el F1-score de la clase desocupada.

### Representación cromosómica

Cada individuo es un **vector binario** de longitud igual al número de variables codificadas.

Ejemplo: `[1, 0, 1, 1, 0, 0, 1, ...]`

- `1` = la variable se utiliza en el modelo
- `0` = la variable no se utiliza

### Función de fitness

```
fitness = F1_desocupados - penalización_por_variables
```

Donde:
- `F1_desocupados`: F1-score promedio de la clase desocupada, calculado mediante **validación cruzada estratificada de 5 folds** sobre el conjunto de entrenamiento.
- `penalización_por_variables`: `0.005 * (n_seleccionadas / n_total)`, una penalización pequeña que favorece modelos más simples sin sacrificar desempeño significativamente.

**Importante:** El fitness se calcula **exclusivamente** sobre el conjunto de entrenamiento. El conjunto de prueba no participa en ningún momento del proceso evolutivo.

### Operadores genéticos

| Componente | Implementación |
|------------|---------------|
| Selección | Torneo (tamaño 3) |
| Cruce | Dos puntos (probabilidad 0.8) |
| Mutación | Bit flip (probabilidad 0.1) |
| Elitismo | 2 mejores individuos pasan directamente |

### Parámetros utilizados

| Parámetro | Valor |
|-----------|-------|
| Tamaño de población | 30 |
| Número de generaciones | 30 |
| Probabilidad de cruce | 0.8 |
| Probabilidad de mutación | 0.1 |
| Tamaño del torneo | 3 |
| Individuos de élite | 2 |
| Penalización por variables | 0.005 |
| Folds de validación cruzada | 5 |

---

## 14. Resultados

> **Nota:** Los resultados presentados a continuación se obtuvieron a partir de la ejecución real del proyecto con los datos del INDEC. No son valores inventados ni estimados.

Los resultados detallados se encuentran en los archivos generados en la carpeta `resultados/metricas/`. Los gráficos se encuentran en `resultados/graficos/`.

Consultar el archivo `resultados/metricas/resultados_completos.json` para las métricas exactas, y `resultados/metricas/comparacion_modelos.csv` para la tabla comparativa.

---

## 15. Variables seleccionadas

El Algoritmo Genético seleccionó un subconjunto de variables que se detalla en el archivo `resultados/metricas/variables_seleccionadas.csv`.

El análisis de importancia de variables del Random Forest se encuentra en `resultados/metricas/importancia_variables_base.csv`.

**Importante:** Las relaciones identificadas por el modelo son de naturaleza predictiva, no causal. El modelo identifica patrones estadísticos útiles para la predicción, pero no establece que una variable "cause" desocupación.

---

## 16. Comparación modelo base vs. optimizado

La comparación detallada entre el modelo base (con todas las variables) y el modelo optimizado (con variables seleccionadas por el AG) se encuentra en `resultados/metricas/comparacion_modelos.csv`.

Ambos modelos fueron evaluados sobre **exactamente el mismo conjunto de prueba**, lo cual garantiza la validez de la comparación.

---

## 17. Análisis de la clase desocupada

La clase desocupada (1) es la clase de mayor interés en este problema porque:

1. Es la clase minoritaria (~5,75% de la PEA).
2. Su correcta identificación es fundamental para la finalidad práctica del modelo.
3. Un **Falso Negativo** (persona desocupada clasificada como ocupada) implica que el sistema no detecta una situación que justamente se intenta identificar.

Por esta razón, se presta especial atención al **Recall** y al **F1-score** de esta clase.

---

## 18. Finalidad práctica

El modelo **no** tiene como finalidad simplemente clasificar personas ni tomar decisiones automáticas.

La finalidad práctica es **identificar perfiles o grupos** que presentan mayor probabilidad de desocupación. A partir de esa información, organismos públicos podrían orientar recursos hacia:

- **Capacitación laboral:** Programas focalizados en habilidades demandadas.
- **Formación profesional:** Oferta educativa alineada con las necesidades del mercado.
- **Orientación laboral:** Asesoramiento para la búsqueda de empleo.
- **Intermediación laboral:** Conexión entre oferta y demanda de trabajo.
- **Programas de inserción:** Iniciativas para facilitar el primer empleo o la reinserción.
- **Recalificación profesional:** Actualización de competencias.
- **Seguimiento preventivo:** Monitoreo de poblaciones en riesgo.

La predicción debe interpretarse como un **indicador de apoyo** para la toma de decisiones. La decisión final corresponde a organismos y responsables humanos.

---

## 19. Limitaciones

1. **El modelo genera predicciones, no certezas.** Toda clasificación tiene un margen de error.
2. **Correlación no implica causalidad.** Las relaciones identificadas son estadísticas, no causales.
3. **Los datos corresponden a un período determinado** (3er trimestre de 2025). Los patrones pueden variar en otros períodos.
4. **La EPH es una encuesta** con características metodológicas específicas (diseño muestral, cobertura geográfica, etc.) que condicionan la generalización de los resultados.
5. **Puede existir sesgo en los datos** derivado del diseño de la encuesta o de las características de la población encuestada.
6. **Variables sensibles:** Las variables sociodemográficas (sexo, edad, nivel educativo) pueden generar riesgos de discriminación si se utilizan incorrectamente en contextos reales.
7. **Desbalance de clases:** La clase desocupada representa aproximadamente el 5,75% de la PEA, lo cual dificulta su detección.
8. **Las variables disponibles** en la EPH pueden no capturar todos los factores relevantes para la empleabilidad (experiencia laboral detallada, habilidades específicas, redes de contacto, etc.).

---

## 20. Consideraciones éticas

- El modelo **no debería utilizarse** para negar derechos, beneficios o acceso a servicios a ninguna persona.
- La decisión final sobre cualquier intervención debe permanecer bajo **supervisión humana**.
- Los resultados no deben interpretarse como juicios de valor sobre las personas.
- Se debe evitar la estigmatización de grupos o perfiles identificados por el modelo.
- El objetivo académico es estudiar la capacidad predictiva de los datos y evaluar la utilidad de los Algoritmos Genéticos para la selección de variables.

---

## 21. Conclusiones

> **Las conclusiones se basan exclusivamente en los resultados obtenidos durante la ejecución del proyecto con datos reales del INDEC.**

Las conclusiones detalladas basadas en los resultados numéricos se incluyen en el notebook `notebooks/analisis_modelo.ipynb` y en los archivos de resultados generados en `resultados/metricas/`.

Puntos generales a evaluar:
1. ¿El modelo pudo predecir la condición de desocupación?
2. ¿Qué desempeño obtuvo sobre la clase desocupada?
3. ¿El Algoritmo Genético mejoró, mantuvo o empeoró el desempeño?
4. ¿Cuántas variables se redujeron?
5. ¿Qué variables fueron seleccionadas como más relevantes?
6. ¿Qué utilidad práctica podría tener el modelo?

Si el Algoritmo Genético no mejoró el modelo, esto también constituye un resultado válido de investigación.

---

## 22. Trabajo futuro

- Explorar otros algoritmos de clasificación (XGBoost, SVM, redes neuronales).
- Incorporar variables laborales adicionales de la EPH.
- Evaluar la optimización conjunta de selección de variables e hiperparámetros.
- Analizar la estabilidad temporal del modelo con datos de otros trimestres.
- Implementar técnicas de sobremuestreo (SMOTE) para abordar el desbalance.
- Desarrollar una interfaz de consulta para facilitar el uso por parte de organismos públicos.
- Realizar análisis por subgrupos (jóvenes, mujeres, regiones específicas).

---

## 23. Fuentes consultadas

1. INDEC - Instituto Nacional de Estadística y Censos. *Encuesta Permanente de Hogares (EPH)*. Microdatos del tercer trimestre de 2025. https://www.indec.gob.ar/indec/web/Institucional-Indec-BasesDeDatos

2. INDEC. *Diseño de registro y estructura para las bases de microdatos*. Tercer trimestre de 2025.

3. Scikit-learn: Machine Learning in Python. Pedregosa et al., JMLR 12, pp. 2825-2830, 2011. https://scikit-learn.org

4. DEAP: Distributed Evolutionary Algorithms in Python. Fortin et al., JMLR, 2012. https://deap.readthedocs.io

5. Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.

6. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
