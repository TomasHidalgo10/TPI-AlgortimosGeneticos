"""
Módulo de evaluación de modelos.
Compara el modelo base con el modelo optimizado
mediante Algoritmo Genético.
"""

import pandas as pd
import numpy as np
from codigo.configuracion import PARAMETROS_RF, SEMILLA
from codigo.modelo_base import crear_modelo_base, entrenar_modelo, evaluar_modelo


def entrenar_modelo_optimizado(X_entrenamiento, y_entrenamiento, X_prueba, y_prueba,
                                indices_seleccionados, nombres_columnas):
    """
    Entrena un modelo Random Forest utilizando SOLO las variables
    seleccionadas por el Algoritmo Genético.

    Parámetros
    ----------
    X_entrenamiento : array-like
        Datos de entrenamiento codificados completos.
    y_entrenamiento : array-like
        Variable objetivo de entrenamiento.
    X_prueba : array-like
        Datos de prueba codificados completos.
    y_prueba : array-like
        Variable objetivo de prueba.
    indices_seleccionados : list
        Índices de las variables seleccionadas por el AG.
    nombres_columnas : list
        Nombres de todas las columnas.

    Retorna
    -------
    dict
        Métricas del modelo optimizado.
    estimator
        Modelo entrenado.
    float
        Tiempo de entrenamiento.
    """
    print("\n" + "=" * 60)
    print("  MODELO OPTIMIZADO (variables del Algoritmo Genético)")
    print("=" * 60)

    # Seleccionar variables del AG
    X_entren_opt = X_entrenamiento[:, indices_seleccionados]
    X_prueba_opt = X_prueba[:, indices_seleccionados]

    variables_usadas = [nombres_columnas[i] for i in indices_seleccionados]
    print(f"\nVariables utilizadas: {len(variables_usadas)} de {len(nombres_columnas)}")

    # Crear y entrenar modelo con mismos parámetros que el base
    modelo_opt = crear_modelo_base()
    modelo_opt, tiempo = entrenar_modelo(modelo_opt, X_entren_opt, y_entrenamiento)

    # Evaluar con EXACTAMENTE el mismo conjunto de prueba
    metricas_opt = evaluar_modelo(modelo_opt, X_prueba_opt, y_prueba,
                                  nombre_modelo="Modelo Optimizado (AG)")

    metricas_opt['tiempo_entrenamiento'] = tiempo
    metricas_opt['n_variables'] = len(variables_usadas)
    metricas_opt['variables_usadas'] = variables_usadas

    return metricas_opt, modelo_opt, tiempo


def comparar_modelos(metricas_base, metricas_optimizado, tiempo_base, tiempo_opt,
                     n_vars_base, n_vars_opt):
    """
    Genera una tabla comparativa entre el modelo base y el optimizado.

    Ambos modelos se evaluaron sobre EXACTAMENTE el mismo conjunto de prueba,
    garantizando la validez de la comparación.

    Parámetros
    ----------
    metricas_base : dict
        Métricas del modelo base.
    metricas_optimizado : dict
        Métricas del modelo optimizado.
    tiempo_base : float
        Tiempo de entrenamiento del modelo base.
    tiempo_opt : float
        Tiempo de entrenamiento del modelo optimizado.
    n_vars_base : int
        Número de variables del modelo base.
    n_vars_opt : int
        Número de variables del modelo optimizado.

    Retorna
    -------
    pandas.DataFrame
        Tabla comparativa.
    """
    filas = [
        ('Accuracy', metricas_base['accuracy'], metricas_optimizado['accuracy']),
        ('Precision (desocupado)', metricas_base['precision_desocupado'],
         metricas_optimizado['precision_desocupado']),
        ('Recall (desocupado)', metricas_base['recall_desocupado'],
         metricas_optimizado['recall_desocupado']),
        ('F1-score (desocupado)', metricas_base['f1_desocupado'],
         metricas_optimizado['f1_desocupado']),
        ('F1 Macro', metricas_base['f1_macro'], metricas_optimizado['f1_macro']),
        ('Precision (ocupado)', metricas_base['precision_ocupado'],
         metricas_optimizado['precision_ocupado']),
        ('Recall (ocupado)', metricas_base['recall_ocupado'],
         metricas_optimizado['recall_ocupado']),
        ('F1-score (ocupado)', metricas_base['f1_ocupado'],
         metricas_optimizado['f1_ocupado']),
        ('Cantidad de variables', n_vars_base, n_vars_opt),
        ('Tiempo entrenamiento (s)', tiempo_base, tiempo_opt),
    ]

    tabla = pd.DataFrame(filas, columns=['Métrica', 'Modelo Base', 'Modelo Optimizado'])
    tabla['Diferencia'] = tabla['Modelo Optimizado'] - tabla['Modelo Base']

    # Formatear para mostrar
    print("\n" + "=" * 80)
    print("  COMPARACIÓN: MODELO BASE vs. MODELO OPTIMIZADO (AG)")
    print("=" * 80)
    print(f"\n{'Métrica':<30} {'Modelo Base':>15} {'Modelo Opt.':>15} {'Diferencia':>15}")
    print("-" * 75)

    for _, fila in tabla.iterrows():
        nombre = fila['Métrica']
        base_val = fila['Modelo Base']
        opt_val = fila['Modelo Optimizado']
        diff = fila['Diferencia']

        if nombre in ['Cantidad de variables', 'Tiempo entrenamiento (s)']:
            if nombre == 'Cantidad de variables':
                print(f"{nombre:<30} {int(base_val):>15} {int(opt_val):>15} {int(diff):>15}")
            else:
                print(f"{nombre:<30} {base_val:>15.2f} {opt_val:>15.2f} {diff:>15.2f}")
        else:
            indicador = '↑' if diff > 0 else ('↓' if diff < 0 else '=')
            print(f"{nombre:<30} {base_val:>15.4f} {opt_val:>15.4f} {diff:>+15.4f} {indicador}")

    print("-" * 75)

    # Análisis automático
    print("\n=== ANÁLISIS DE LA COMPARACIÓN ===")

    diff_f1 = metricas_optimizado['f1_desocupado'] - metricas_base['f1_desocupado']
    diff_recall = metricas_optimizado['recall_desocupado'] - metricas_base['recall_desocupado']
    reduccion_vars = (1 - n_vars_opt / n_vars_base) * 100

    if diff_f1 > 0.01:
        print(f"  ✓ El AG MEJORÓ el F1 de desocupados en {diff_f1:+.4f}")
    elif diff_f1 > -0.01:
        print(f"  ≈ El AG MANTUVO el F1 de desocupados (diferencia: {diff_f1:+.4f})")
    else:
        print(f"  ✗ El AG REDUJO el F1 de desocupados en {diff_f1:+.4f}")

    if diff_recall > 0.01:
        print(f"  ✓ El AG MEJORÓ el Recall de desocupados en {diff_recall:+.4f}")
    elif diff_recall > -0.01:
        print(f"  ≈ El AG MANTUVO el Recall de desocupados (diferencia: {diff_recall:+.4f})")
    else:
        print(f"  ✗ El AG REDUJO el Recall de desocupados en {diff_recall:+.4f}")

    print(f"  📊 Reducción de variables: {reduccion_vars:.1f}% ({n_vars_base} → {n_vars_opt})")

    if reduccion_vars > 0 and diff_f1 >= -0.01:
        print(f"\n  → El Algoritmo Genético logró reducir la dimensionalidad")
        print(f"    manteniendo un desempeño comparable, lo cual valida")
        print(f"    su utilidad para la selección de variables.")
    elif diff_f1 > 0.01:
        print(f"\n  → El Algoritmo Genético logró mejorar el desempeño del modelo")
        print(f"    al seleccionar un subconjunto más relevante de variables.")

    return tabla


def generar_tabla_variables_seleccionadas(variables_seleccionadas, nombres_columnas,
                                          variables_predictoras_info):
    """
    Genera una tabla con las variables seleccionadas por el AG
    y sus descripciones según la documentación oficial del INDEC.

    Parámetros
    ----------
    variables_seleccionadas : list
        Nombres de las variables seleccionadas.
    nombres_columnas : list
        Todos los nombres de columnas codificadas.
    variables_predictoras_info : dict
        Información de las variables desde configuracion.py.

    Retorna
    -------
    pandas.DataFrame
        Tabla con información de cada variable.
    """
    filas = []
    for col in nombres_columnas:
        seleccionada = "Sí" if col in variables_seleccionadas else "No"

        # Determinar la variable original
        var_original = col.split('_')[0] if '_' in col else col
        # Para variables con formato AGLOMERADO_X, CH03_X, etc.
        for var_nombre in variables_predictoras_info:
            if col.startswith(var_nombre):
                var_original = var_nombre
                break

        descripcion = variables_predictoras_info.get(var_original, {}).get(
            'descripcion', col
        )
        tipo = variables_predictoras_info.get(var_original, {}).get('tipo', 'codificada')

        filas.append({
            'Variable codificada': col,
            'Variable original': var_original,
            'Seleccionada': seleccionada,
            'Descripción': descripcion,
            'Tipo': tipo
        })

    tabla = pd.DataFrame(filas)

    # Resumen por variable original
    print("\n=== RESUMEN DE SELECCIÓN POR VARIABLE ORIGINAL ===")
    for var_nombre, info in variables_predictoras_info.items():
        cols_var = [c for c in nombres_columnas if c.startswith(var_nombre)]
        cols_sel = [c for c in variables_seleccionadas if c.startswith(var_nombre)]
        if cols_var:
            print(f"  {var_nombre} ({info['descripcion']}): "
                  f"{len(cols_sel)}/{len(cols_var)} categorías seleccionadas")

    return tabla
