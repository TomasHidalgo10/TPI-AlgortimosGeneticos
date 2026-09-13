"""
Módulo de visualizaciones.
Genera todos los gráficos requeridos para el Trabajo Práctico,
incluyendo distribuciones, evolución del AG, comparaciones
y matrices de confusión.
"""

import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para guardar sin mostrar
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from codigo.configuracion import COLORES, TAMANO_FIGURA, DPI_GRAFICOS, RUTA_GRAFICOS, FORMATO_GRAFICOS


def configurar_estilo():
    """Configura el estilo global de los gráficos."""
    plt.rcParams.update({
        'figure.figsize': TAMANO_FIGURA,
        'figure.dpi': DPI_GRAFICOS,
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.titleweight': 'bold',
        'figure.facecolor': 'white',
        'axes.facecolor': '#fafafa',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    sns.set_style("whitegrid")


def guardar_grafico(nombre_archivo):
    """Guarda el gráfico actual en la carpeta de resultados."""
    os.makedirs(RUTA_GRAFICOS, exist_ok=True)
    ruta = os.path.join(RUTA_GRAFICOS, f"{nombre_archivo}.{FORMATO_GRAFICOS}")
    plt.savefig(ruta, dpi=DPI_GRAFICOS, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Gráfico guardado: {ruta}")
    return ruta


def grafico_distribucion_clases(y, titulo="Distribución de la condición laboral"):
    """
    Genera un gráfico de barras con la distribución de ocupados y desocupados.

    Parámetros
    ----------
    y : array-like
        Variable objetivo (0 = ocupado, 1 = desocupado).
    titulo : str
        Título del gráfico.
    """
    configurar_estilo()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico de barras
    etiquetas = ['Ocupado (0)', 'Desocupado (1)']
    valores = [int((np.array(y) == 0).sum()), int((np.array(y) == 1).sum())]
    colores_barras = [COLORES['ocupado'], COLORES['desocupado']]

    barras = axes[0].bar(etiquetas, valores, color=colores_barras, edgecolor='white', linewidth=1.5)
    axes[0].set_title(titulo)
    axes[0].set_ylabel('Cantidad de registros')

    # Añadir valores sobre las barras
    for barra, valor in zip(barras, valores):
        axes[0].text(barra.get_x() + barra.get_width()/2., barra.get_height() + 100,
                     f'{valor:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Gráfico circular
    porcentajes = [v / sum(valores) * 100 for v in valores]
    axes[1].pie(valores, labels=[f'{e}\n({p:.1f}%)' for e, p in zip(etiquetas, porcentajes)],
                colors=colores_barras, autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 11}, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Proporción de clases')

    plt.suptitle('Distribución de la variable objetivo en la PEA',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    return guardar_grafico('01_distribucion_clases')


def grafico_evolucion_fitness(registro_evolucion):
    """
    Genera el gráfico de evolución del fitness del Algoritmo Genético.

    Parámetros
    ----------
    registro_evolucion : dict
        Diccionario con datos de la evolución por generación.
    """
    configurar_estilo()
    fig, ax1 = plt.subplots(figsize=(12, 6))

    generaciones = registro_evolucion['generacion']
    mejor_fitness = registro_evolucion['mejor_fitness']
    fitness_promedio = registro_evolucion['fitness_promedio']

    # Fitness
    ax1.plot(generaciones, mejor_fitness, 'o-', color=COLORES['fitness'],
             linewidth=2, markersize=4, label='Mejor fitness', zorder=3)
    ax1.plot(generaciones, fitness_promedio, 's--', color='#7f8c8d',
             linewidth=1.5, markersize=3, alpha=0.7, label='Fitness promedio')

    ax1.fill_between(generaciones, fitness_promedio, mejor_fitness,
                     alpha=0.15, color=COLORES['fitness'])

    ax1.set_xlabel('Generación')
    ax1.set_ylabel('F1-score (clase desocupada)')
    ax1.set_title('Evolución del fitness del Algoritmo Genético')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    return guardar_grafico('02_evolucion_fitness')


def grafico_evolucion_variables(registro_evolucion):
    """
    Genera el gráfico de evolución de la cantidad de variables seleccionadas.

    Parámetros
    ----------
    registro_evolucion : dict
        Diccionario con datos de la evolución por generación.
    """
    configurar_estilo()
    fig, ax = plt.subplots(figsize=(12, 5))

    generaciones = registro_evolucion['generacion']
    n_variables = registro_evolucion['n_variables_mejor']

    ax.plot(generaciones, n_variables, 'D-', color=COLORES['modelo_opt'],
            linewidth=2, markersize=5)
    ax.fill_between(generaciones, n_variables, alpha=0.2, color=COLORES['modelo_opt'])

    ax.set_xlabel('Generación')
    ax.set_ylabel('Cantidad de variables seleccionadas')
    ax.set_title('Evolución de la cantidad de variables seleccionadas por generación')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return guardar_grafico('03_evolucion_variables')


def grafico_comparacion_metricas(metricas_base, metricas_opt):
    """
    Genera un gráfico comparativo de métricas entre ambos modelos.

    Parámetros
    ----------
    metricas_base : dict
        Métricas del modelo base.
    metricas_opt : dict
        Métricas del modelo optimizado.
    """
    configurar_estilo()
    fig, ax = plt.subplots(figsize=(12, 6))

    metricas_nombres = [
        'Accuracy', 'Precision\n(desocupado)', 'Recall\n(desocupado)',
        'F1-score\n(desocupado)', 'F1 Macro'
    ]
    valores_base = [
        metricas_base['accuracy'], metricas_base['precision_desocupado'],
        metricas_base['recall_desocupado'], metricas_base['f1_desocupado'],
        metricas_base['f1_macro']
    ]
    valores_opt = [
        metricas_opt['accuracy'], metricas_opt['precision_desocupado'],
        metricas_opt['recall_desocupado'], metricas_opt['f1_desocupado'],
        metricas_opt['f1_macro']
    ]

    x = np.arange(len(metricas_nombres))
    ancho = 0.35

    barras1 = ax.bar(x - ancho/2, valores_base, ancho, label='Modelo Base',
                     color=COLORES['modelo_base'], edgecolor='white', linewidth=1.5)
    barras2 = ax.bar(x + ancho/2, valores_opt, ancho, label='Modelo Optimizado (AG)',
                     color=COLORES['modelo_opt'], edgecolor='white', linewidth=1.5)

    # Valores sobre barras
    for barra in barras1:
        h = barra.get_height()
        ax.text(barra.get_x() + barra.get_width()/2., h + 0.005,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for barra in barras2:
        h = barra.get_height()
        ax.text(barra.get_x() + barra.get_width()/2., h + 0.005,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Valor de la métrica')
    ax.set_title('Comparación de métricas: Modelo Base vs. Modelo Optimizado (AG)')
    ax.set_xticks(x)
    ax.set_xticklabels(metricas_nombres)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return guardar_grafico('04_comparacion_metricas')


def grafico_matriz_confusion(matriz, nombre_modelo, nombre_archivo):
    """
    Genera una visualización de la matriz de confusión.

    Parámetros
    ----------
    matriz : array-like
        Matriz de confusión 2x2.
    nombre_modelo : str
        Nombre del modelo para el título.
    nombre_archivo : str
        Nombre del archivo de salida.
    """
    configurar_estilo()
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ocupado (0)', 'Desocupado (1)'],
                yticklabels=['Ocupado (0)', 'Desocupado (1)'],
                linewidths=2, linecolor='white',
                annot_kws={'size': 16, 'fontweight': 'bold'},
                ax=ax)

    ax.set_xlabel('Clase predicha', fontsize=12)
    ax.set_ylabel('Clase real', fontsize=12)
    ax.set_title(f'Matriz de confusión - {nombre_modelo}', fontsize=14, fontweight='bold')

    # Añadir interpretación
    vn, fp = matriz[0]
    fn, vp = matriz[1]
    texto = (f'VP={vp} | FN={fn} | FP={fp} | VN={vn}\n'
             f'Recall desocupado: {vp/(vp+fn):.3f}' if (vp+fn) > 0 else '')
    ax.text(0.5, -0.15, texto, transform=ax.transAxes,
            ha='center', fontsize=10, style='italic', color='#555555')

    plt.tight_layout()
    return guardar_grafico(nombre_archivo)


def grafico_importancia_variables(importancias, top_n=20):
    """
    Genera un gráfico de importancia de variables del Random Forest.

    Parámetros
    ----------
    importancias : dict
        Diccionario {nombre_variable: importancia} ordenado.
    top_n : int
        Cantidad de variables a mostrar.
    """
    configurar_estilo()
    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.35)))

    # Tomar top N
    nombres = list(importancias.keys())[:top_n]
    valores = list(importancias.values())[:top_n]

    # Invertir para que la más importante esté arriba
    nombres = nombres[::-1]
    valores = valores[::-1]

    colores_barras = [COLORES['modelo_base']] * len(nombres)

    ax.barh(range(len(nombres)), valores, color=colores_barras,
            edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(nombres)))
    ax.set_yticklabels(nombres, fontsize=9)
    ax.set_xlabel('Importancia (Gini)')
    ax.set_title(f'Importancia de las variables (Top {top_n}) - Modelo Base')
    ax.grid(True, alpha=0.3, axis='x')

    # Valores al final de las barras
    for i, v in enumerate(valores):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=8)

    plt.tight_layout()
    return guardar_grafico('07_importancia_variables')


def grafico_analisis_exploratorio(datos_pea):
    """
    Genera gráficos de análisis exploratorio adicionales.

    Parámetros
    ----------
    datos_pea : pandas.DataFrame
        DataFrame de la PEA con la variable objetivo.
    """
    configurar_estilo()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Distribución por sexo y condición laboral
    ax = axes[0, 0]
    sexo_target = datos_pea.groupby(['CH04', 'desocupado']).size().unstack(fill_value=0)
    sexo_labels = {1: 'Varón', 2: 'Mujer'}
    sexo_target.index = [sexo_labels.get(x, x) for x in sexo_target.index]
    sexo_target.columns = ['Ocupado', 'Desocupado']
    sexo_target.plot(kind='bar', ax=ax, color=[COLORES['ocupado'], COLORES['desocupado']],
                     edgecolor='white', linewidth=1)
    ax.set_title('Condición laboral por sexo')
    ax.set_xlabel('Sexo')
    ax.set_ylabel('Cantidad')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend()

    # 2. Distribución de edad por condición laboral
    ax = axes[0, 1]
    edad_ocup = datos_pea[datos_pea['desocupado'] == 0]['CH06']
    edad_desoc = datos_pea[datos_pea['desocupado'] == 1]['CH06']
    ax.hist(edad_ocup, bins=30, alpha=0.6, color=COLORES['ocupado'],
            label='Ocupado', density=True, edgecolor='white')
    ax.hist(edad_desoc, bins=30, alpha=0.6, color=COLORES['desocupado'],
            label='Desocupado', density=True, edgecolor='white')
    ax.set_title('Distribución de edad por condición laboral')
    ax.set_xlabel('Edad')
    ax.set_ylabel('Densidad')
    ax.legend()

    # 3. Nivel educativo y condición laboral
    ax = axes[1, 0]
    nivel_labels = {1: 'Prim. Inc.', 2: 'Prim. Comp.', 3: 'Sec. Inc.',
                    4: 'Sec. Comp.', 5: 'Sup. Inc.', 6: 'Sup. Comp.', 7: 'Sin Instr.'}
    nivel_target = datos_pea.groupby(['NIVEL_ED', 'desocupado']).size().unstack(fill_value=0)
    nivel_target.index = [nivel_labels.get(x, x) for x in nivel_target.index]
    nivel_target.columns = ['Ocupado', 'Desocupado']
    # Calcular tasa de desocupación por nivel
    nivel_target['Tasa desocupación (%)'] = (
        nivel_target['Desocupado'] / (nivel_target['Ocupado'] + nivel_target['Desocupado']) * 100
    )
    nivel_target['Tasa desocupación (%)'].plot(
        kind='bar', ax=ax, color=COLORES['desocupado'],
        edgecolor='white', linewidth=1
    )
    ax.set_title('Tasa de desocupación por nivel educativo')
    ax.set_xlabel('Nivel educativo')
    ax.set_ylabel('Tasa de desocupación (%)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # 4. Estado civil y condición laboral
    ax = axes[1, 1]
    civil_labels = {1: 'Unido/a', 2: 'Casado/a', 3: 'Sep./Div.',
                    4: 'Viudo/a', 5: 'Soltero/a'}
    civil_target = datos_pea.groupby(['CH07', 'desocupado']).size().unstack(fill_value=0)
    civil_target.index = [civil_labels.get(x, x) for x in civil_target.index]
    civil_target.columns = ['Ocupado', 'Desocupado']
    civil_target['Tasa desocupación (%)'] = (
        civil_target['Desocupado'] / (civil_target['Ocupado'] + civil_target['Desocupado']) * 100
    )
    civil_target['Tasa desocupación (%)'].plot(
        kind='bar', ax=ax, color=COLORES['fitness'],
        edgecolor='white', linewidth=1
    )
    ax.set_title('Tasa de desocupación por estado civil')
    ax.set_xlabel('Estado civil')
    ax.set_ylabel('Tasa de desocupación (%)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.suptitle('Análisis exploratorio de variables sociodemográficas',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    return guardar_grafico('08_analisis_exploratorio')


def generar_todas_las_visualizaciones(datos_pea, y_objetivo, registro_evolucion,
                                       metricas_base, metricas_opt,
                                       importancias_base):
    """
    Genera todas las visualizaciones obligatorias del TP.

    Parámetros
    ----------
    datos_pea : pandas.DataFrame
        DataFrame de la PEA con variables originales.
    y_objetivo : array-like
        Variable objetivo completa.
    registro_evolucion : dict
        Registro de la evolución del AG.
    metricas_base : dict
        Métricas del modelo base.
    metricas_opt : dict
        Métricas del modelo optimizado.
    importancias_base : dict
        Importancia de variables del modelo base.

    Retorna
    -------
    list
        Lista de rutas de los gráficos generados.
    """
    print("\n" + "=" * 60)
    print("  GENERANDO VISUALIZACIONES")
    print("=" * 60)

    rutas = []

    # 1. Distribución de clases
    print("\n1. Distribución de clases...")
    rutas.append(grafico_distribucion_clases(y_objetivo))

    # 2. Evolución del fitness
    print("2. Evolución del fitness...")
    rutas.append(grafico_evolucion_fitness(registro_evolucion))

    # 3. Evolución de variables seleccionadas
    print("3. Evolución de variables seleccionadas...")
    rutas.append(grafico_evolucion_variables(registro_evolucion))

    # 4. Comparación de métricas
    print("4. Comparación de métricas...")
    rutas.append(grafico_comparacion_metricas(metricas_base, metricas_opt))

    # 5. Matriz de confusión - Modelo base
    print("5. Matriz de confusión - Modelo Base...")
    rutas.append(grafico_matriz_confusion(
        metricas_base['matriz_confusion'], 'Modelo Base', '05_matriz_confusion_base'
    ))

    # 6. Matriz de confusión - Modelo optimizado
    print("6. Matriz de confusión - Modelo Optimizado...")
    rutas.append(grafico_matriz_confusion(
        metricas_opt['matriz_confusion'], 'Modelo Optimizado (AG)',
        '06_matriz_confusion_optimizado'
    ))

    # 7. Importancia de variables
    print("7. Importancia de variables...")
    rutas.append(grafico_importancia_variables(importancias_base))

    # 8. Análisis exploratorio
    print("8. Análisis exploratorio...")
    rutas.append(grafico_analisis_exploratorio(datos_pea))

    print(f"\n  ✓ Se generaron {len(rutas)} gráficos en: {RUTA_GRAFICOS}")
    return rutas
