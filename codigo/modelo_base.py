"""
Módulo del modelo base.
Implementa el modelo de Random Forest como línea de base
para la comparación con el modelo optimizado.
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from codigo.configuracion import PARAMETROS_RF, SEMILLA


def crear_modelo_base():
    """
    Crea el modelo Random Forest con los parámetros de configuracion.py.
    Usa class_weight='balanced' para compensar el desbalance de clases.

    Retorna
    -------
    RandomForestClassifier
    """
    modelo = RandomForestClassifier(**PARAMETROS_RF)

    print("=== MODELO BASE: Random Forest ===")
    print("Parámetros:")
    for parametro, valor in PARAMETROS_RF.items():
        print(f"  {parametro}: {valor}")
    print("\n  → Se utiliza class_weight='balanced' para compensar")
    print("    el desbalance de clases (muchos más ocupados que desocupados)")

    return modelo


def entrenar_modelo(modelo, X_entrenamiento, y_entrenamiento):
    """
    Entrena el modelo con los datos de entrenamiento.

    Parámetros
    ----------
    modelo : estimator
        Modelo de scikit-learn.
    X_entrenamiento : array-like
        Variables predictoras de entrenamiento.
    y_entrenamiento : array-like
        Variable objetivo de entrenamiento.

    Retorna
    -------
    estimator
        Modelo entrenado.
    float
        Tiempo de entrenamiento en segundos.
    """
    print("\nEntrenando modelo...")
    inicio = time.time()
    modelo.fit(X_entrenamiento, y_entrenamiento)
    tiempo_entrenamiento = time.time() - inicio
    print(f"Modelo entrenado en {tiempo_entrenamiento:.2f} segundos")

    return modelo, tiempo_entrenamiento


def evaluar_modelo(modelo, X_prueba, y_prueba, nombre_modelo="Modelo"):
    """
    Evalúa el modelo sobre el conjunto de prueba.
    Prioriza métricas de la clase desocupada (pos_label=1).

    Parámetros
    ----------
    modelo : estimator
    X_prueba, y_prueba : array-like
    nombre_modelo : str

    Retorna
    -------
    dict
    """
    predicciones = modelo.predict(X_prueba)

    # Métricas generales
    accuracy = accuracy_score(y_prueba, predicciones)

    # Métricas para clase desocupada (pos_label=1)
    precision_desocupado = precision_score(y_prueba, predicciones, pos_label=1, zero_division=0)
    recall_desocupado = recall_score(y_prueba, predicciones, pos_label=1, zero_division=0)
    f1_desocupado = f1_score(y_prueba, predicciones, pos_label=1, zero_division=0)

    # Métricas para clase ocupada (pos_label=0)
    precision_ocupado = precision_score(y_prueba, predicciones, pos_label=0, zero_division=0)
    recall_ocupado = recall_score(y_prueba, predicciones, pos_label=0, zero_division=0)
    f1_ocupado = f1_score(y_prueba, predicciones, pos_label=0, zero_division=0)

    # F1 macro (promedio no ponderado)
    f1_macro = f1_score(y_prueba, predicciones, average='macro', zero_division=0)

    # Matriz de confusión
    matriz = confusion_matrix(y_prueba, predicciones)

    metricas = {
        'nombre_modelo': nombre_modelo,
        'accuracy': accuracy,
        'precision_desocupado': precision_desocupado,
        'recall_desocupado': recall_desocupado,
        'f1_desocupado': f1_desocupado,
        'precision_ocupado': precision_ocupado,
        'recall_ocupado': recall_ocupado,
        'f1_ocupado': f1_ocupado,
        'f1_macro': f1_macro,
        'matriz_confusion': matriz,
        'predicciones': predicciones,
        'reporte': classification_report(y_prueba, predicciones,
                                          target_names=['Ocupado (0)', 'Desocupado (1)'],
                                          zero_division=0)
    }

    # Imprimir resultados
    print(f"\n{'='*60}")
    print(f"  EVALUACIÓN: {nombre_modelo}")
    print(f"{'='*60}")
    print(f"\n  Accuracy: {accuracy:.4f}")
    print(f"\n  --- Clase DESOCUPADO (1) [clase de interés] ---")
    print(f"  Precision: {precision_desocupado:.4f}")
    print(f"  Recall:    {recall_desocupado:.4f}")
    print(f"  F1-score:  {f1_desocupado:.4f}")
    print(f"\n  --- Clase OCUPADO (0) ---")
    print(f"  Precision: {precision_ocupado:.4f}")
    print(f"  Recall:    {recall_ocupado:.4f}")
    print(f"  F1-score:  {f1_ocupado:.4f}")
    print(f"\n  F1 Macro: {f1_macro:.4f}")

    print(f"\n  Matriz de confusión:")
    print(f"                     Predicho")
    print(f"                   Ocup.  Desoc.")
    print(f"  Real Ocupado   [{matriz[0][0]:>6}  {matriz[0][1]:>6}]")
    print(f"  Real Desocupado[{matriz[1][0]:>6}  {matriz[1][1]:>6}]")

    vp = matriz[1][1]  # TP
    fn = matriz[1][0]  # FN
    fp = matriz[0][1]  # FP
    vn = matriz[0][0]  # TN

    print(f"\n  Interpretación:")
    print(f"  VP (desocupado → desocupado): {vp}")
    print(f"  FN (desocupado → ocupado):    {fn}  ← especialmente importante")
    print(f"  FP (ocupado → desocupado):    {fp}")
    print(f"  VN (ocupado → ocupado):       {vn}")

    if fn > 0:
        print(f"\n  ⚠ {fn} personas realmente desocupadas fueron clasificadas")
        print(f"    como ocupadas. Esto significa que el modelo no detectó")
        print(f"    su situación de desocupación.")

    print(f"\n{metricas['reporte']}")

    return metricas


def obtener_importancia_variables(modelo, nombres_columnas):
    """
    Obtiene la importancia de las variables del modelo Random Forest.

    Parámetros
    ----------
    modelo : RandomForestClassifier
        Modelo entrenado.
    nombres_columnas : list
        Nombres de las columnas.

    Retorna
    -------
    dict
        Diccionario con nombres de variables y su importancia, ordenado.
    """
    importancias = modelo.feature_importances_
    importancia_dict = dict(zip(nombres_columnas, importancias))

    # Ordenar por importancia descendente
    importancia_ordenada = dict(
        sorted(importancia_dict.items(), key=lambda x: x[1], reverse=True)
    )

    print("\n=== IMPORTANCIA DE VARIABLES (Top 15) ===")
    for i, (nombre, imp) in enumerate(importancia_ordenada.items()):
        if i >= 15:
            break
        barra = '█' * int(imp * 100)
        print(f"  {nombre:<30} {imp:.4f} {barra}")

    return importancia_ordenada
