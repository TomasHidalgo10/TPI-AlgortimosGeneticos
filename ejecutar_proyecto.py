"""
=============================================================================
Script principal de ejecución del Trabajo Práctico Integrador.

"Predicción de la condición de desocupación en la población económicamente
activa de Argentina mediante Machine Learning con selección de variables
optimizada por Algoritmos Genéticos"

Materia: Algoritmos Genéticos
Carrera: Ingeniería en Sistemas de Información

Fuente de datos: INDEC - EPH Total Urbano - 3er Trimestre 2025
=============================================================================

Este script ejecuta el flujo completo del proyecto:
1. Carga de datos
2. Filtrado de la PEA
3. Creación del target
4. Limpieza y preprocesamiento
5. Entrenamiento del modelo base
6. Algoritmo Genético (selección de variables)
7. Entrenamiento del modelo optimizado
8. Comparación de modelos
9. Generación de visualizaciones
10. Guardado de resultados
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from codigo.configuracion import (
    RUTA_METRICAS, RUTA_MODELOS, RUTA_GRAFICOS,
    VARIABLES_PREDICTORAS, SEMILLA
)
from codigo.carga_datos import (
    cargar_datos_personas, filtrar_pea, crear_target, mostrar_info_variables
)
from codigo.preprocesamiento_datos import ejecutar_preprocesamiento_completo
from codigo.modelo_base import (
    crear_modelo_base, entrenar_modelo, evaluar_modelo, obtener_importancia_variables
)
from codigo.algoritmo_genetico import ejecutar_algoritmo_genetico
from codigo.evaluacion_modelo import (
    entrenar_modelo_optimizado, comparar_modelos, generar_tabla_variables_seleccionadas
)
from codigo.visualizaciones import generar_todas_las_visualizaciones


def main():
    """Función principal que ejecuta todo el flujo del proyecto."""

    print("=" * 70)
    print("  TRABAJO PRÁCTICO INTEGRADOR - ALGORITMOS GENÉTICOS")
    print("  Predicción de desocupación en la PEA de Argentina")
    print("  Fuente: INDEC - EPH Total Urbano - 3er Trimestre 2025")
    print("=" * 70)

    inicio_total = time.time()

    # ================================================================
    # PASO 1: Cargar datos
    # ================================================================
    print("\n" + "=" * 60)
    print("  PASO 1: CARGA DE DATOS")
    print("=" * 60)

    datos = cargar_datos_personas()

    # ================================================================
    # PASO 2: Filtrar PEA
    # ================================================================
    print("\n" + "=" * 60)
    print("  PASO 2: FILTRADO DE LA PEA")
    print("=" * 60)

    datos_pea, estadisticas_filtrado = filtrar_pea(datos)

    # ================================================================
    # PASO 3: Crear target
    # ================================================================
    print("\n" + "=" * 60)
    print("  PASO 3: DEFINICIÓN DEL TARGET")
    print("=" * 60)

    datos_pea = crear_target(datos_pea)

    # Mostrar información de variables
    mostrar_info_variables(datos_pea)

    # ================================================================
    # PASO 4: Preprocesamiento completo
    # ================================================================
    print("\n" + "=" * 60)
    print("  PASO 4: PREPROCESAMIENTO")
    print("=" * 60)

    resultado_preprocesamiento = ejecutar_preprocesamiento_completo(datos_pea)

    X_entrenamiento = resultado_preprocesamiento['X_entrenamiento']
    X_prueba = resultado_preprocesamiento['X_prueba']
    y_entrenamiento = resultado_preprocesamiento['y_entrenamiento']
    y_prueba = resultado_preprocesamiento['y_prueba']
    nombres_columnas = resultado_preprocesamiento['nombres_columnas']
    datos_limpios = resultado_preprocesamiento['datos_limpios']

    print(f"\nDimensiones finales:")
    print(f"  X_entrenamiento: {X_entrenamiento.shape}")
    print(f"  X_prueba: {X_prueba.shape}")
    print(f"  Variables codificadas: {len(nombres_columnas)}")
    print(f"  Nombres: {nombres_columnas[:10]}...")

    # ================================================================
    # PASO 5: Modelo Base
    # ================================================================
    print("\n" + "=" * 60)
    print("  PASO 5: MODELO BASE (Random Forest)")
    print("=" * 60)

    modelo_base = crear_modelo_base()
    modelo_base, tiempo_base = entrenar_modelo(modelo_base, X_entrenamiento, y_entrenamiento)
    metricas_base = evaluar_modelo(modelo_base, X_prueba, y_prueba,
                                   nombre_modelo="Modelo Base (todas las variables)")
    metricas_base['tiempo_entrenamiento'] = tiempo_base
    metricas_base['n_variables'] = X_entrenamiento.shape[1]

    # Importancia de variables del modelo base
    importancias_base = obtener_importancia_variables(modelo_base, nombres_columnas)

    # ================================================================
    # PASO 6: Algoritmo Genético
    # ================================================================
    print("\n" + "=" * 60)
    print("  PASO 6: ALGORITMO GENÉTICO")
    print("=" * 60)

    resultado_ag = ejecutar_algoritmo_genetico(
        X_entrenamiento, y_entrenamiento, nombres_columnas
    )

    # ================================================================
    # PASO 7: Modelo Optimizado
    # ================================================================
    print("\n" + "=" * 60)
    print("  PASO 7: MODELO OPTIMIZADO")
    print("=" * 60)

    metricas_opt, modelo_opt, tiempo_opt = entrenar_modelo_optimizado(
        X_entrenamiento, y_entrenamiento, X_prueba, y_prueba,
        resultado_ag['indices_seleccionados'], nombres_columnas
    )

    # Importancia de variables del modelo optimizado
    importancias_opt = obtener_importancia_variables(
        modelo_opt, resultado_ag['variables_seleccionadas']
    )

    # ================================================================
    # PASO 8: Comparación
    # ================================================================
    print("\n" + "=" * 60)
    print("  PASO 8: COMPARACIÓN DE MODELOS")
    print("=" * 60)

    tabla_comparacion = comparar_modelos(
        metricas_base, metricas_opt,
        tiempo_base, tiempo_opt,
        X_entrenamiento.shape[1], resultado_ag['n_variables_seleccionadas']
    )

    # Tabla de variables seleccionadas
    tabla_variables = generar_tabla_variables_seleccionadas(
        resultado_ag['variables_seleccionadas'],
        nombres_columnas,
        VARIABLES_PREDICTORAS
    )

    # ================================================================
    # PASO 9: Visualizaciones
    # ================================================================
    print("\n" + "=" * 60)
    print("  PASO 9: VISUALIZACIONES")
    print("=" * 60)

    rutas_graficos = generar_todas_las_visualizaciones(
        datos_limpios,
        resultado_preprocesamiento['y_entrenamiento'].tolist() +
        resultado_preprocesamiento['y_prueba'].tolist(),
        resultado_ag['registro_evolucion'],
        metricas_base, metricas_opt,
        importancias_base
    )

    # ================================================================
    # PASO 10: Guardar resultados
    # ================================================================
    print("\n" + "=" * 60)
    print("  PASO 10: GUARDADO DE RESULTADOS")
    print("=" * 60)

    guardar_resultados(
        metricas_base, metricas_opt, resultado_ag,
        tabla_comparacion, tabla_variables,
        estadisticas_filtrado, importancias_base, importancias_opt
    )

    # ================================================================
    # RESUMEN FINAL
    # ================================================================
    tiempo_total = time.time() - inicio_total

    print("\n" + "=" * 70)
    print("  EJECUCIÓN COMPLETADA")
    print("=" * 70)
    print(f"\n  Tiempo total: {tiempo_total:.1f} segundos ({tiempo_total/60:.1f} minutos)")
    print(f"\n  Resultados guardados en:")
    print(f"    Gráficos: {RUTA_GRAFICOS}")
    print(f"    Métricas: {RUTA_METRICAS}")
    print(f"\n  Resumen rápido:")
    print(f"    F1 desocupado (base):       {metricas_base['f1_desocupado']:.4f}")
    print(f"    F1 desocupado (optimizado):  {metricas_opt['f1_desocupado']:.4f}")
    print(f"    Recall desocupado (base):    {metricas_base['recall_desocupado']:.4f}")
    print(f"    Recall desocupado (opt):     {metricas_opt['recall_desocupado']:.4f}")
    print(f"    Variables (base):            {X_entrenamiento.shape[1]}")
    print(f"    Variables (optimizado):      {resultado_ag['n_variables_seleccionadas']}")
    print("=" * 70)


def guardar_resultados(metricas_base, metricas_opt, resultado_ag,
                        tabla_comparacion, tabla_variables,
                        estadisticas_filtrado, importancias_base, importancias_opt):
    """Guarda todos los resultados en archivos."""

    os.makedirs(RUTA_METRICAS, exist_ok=True)
    os.makedirs(RUTA_MODELOS, exist_ok=True)

    # 1. Tabla comparativa
    ruta_comparacion = os.path.join(RUTA_METRICAS, 'comparacion_modelos.csv')
    tabla_comparacion.to_csv(ruta_comparacion, index=False, encoding='utf-8-sig')
    print(f"  [OK] Comparación guardada: {ruta_comparacion}")
    
    # 2. Tabla de variables seleccionadas
    ruta_variables = os.path.join(RUTA_METRICAS, 'variables_seleccionadas.csv')
    tabla_variables.to_csv(ruta_variables, index=False, encoding='utf-8-sig')
    print(f"  [OK] Variables seleccionadas: {ruta_variables}")
    
    # 3. Métricas en JSON
    metricas_json = {
        'modelo_base': {
            'accuracy': metricas_base['accuracy'],
            'precision_desocupado': metricas_base['precision_desocupado'],
            'recall_desocupado': metricas_base['recall_desocupado'],
            'f1_desocupado': metricas_base['f1_desocupado'],
            'f1_macro': metricas_base['f1_macro'],
            'n_variables': metricas_base['n_variables'],
            'tiempo_entrenamiento': metricas_base['tiempo_entrenamiento'],
            'matriz_confusion': metricas_base['matriz_confusion'].tolist()
        },
        'modelo_optimizado': {
            'accuracy': metricas_opt['accuracy'],
            'precision_desocupado': metricas_opt['precision_desocupado'],
            'recall_desocupado': metricas_opt['recall_desocupado'],
            'f1_desocupado': metricas_opt['f1_desocupado'],
            'f1_macro': metricas_opt['f1_macro'],
            'n_variables': metricas_opt['n_variables'],
            'tiempo_entrenamiento': metricas_opt['tiempo_entrenamiento'],
            'matriz_confusion': metricas_opt['matriz_confusion'].tolist()
        },
        'algoritmo_genetico': {
            'mejor_fitness': resultado_ag['mejor_fitness'],
            'n_variables_total': resultado_ag['n_variables_total'],
            'n_variables_seleccionadas': resultado_ag['n_variables_seleccionadas'],
            'variables_seleccionadas': resultado_ag['variables_seleccionadas'],
            'tiempo_evolucion': resultado_ag['tiempo_evolucion'],
            'parametros': resultado_ag['parametros_ag']
        },
        'datos': {
            'total_original': estadisticas_filtrado['total_original'],
            'total_pea': estadisticas_filtrado['total_pea'],
            'ocupados': estadisticas_filtrado['cantidad_ocupados'],
            'desocupados': estadisticas_filtrado['cantidad_desocupados'],
        }
    }
    
    ruta_metricas_json = os.path.join(RUTA_METRICAS, 'resultados_completos.json')
    with open(ruta_metricas_json, 'w', encoding='utf-8') as f:
        json.dump(metricas_json, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Métricas JSON: {ruta_metricas_json}")
    
    # 4. Evolución del AG
    evolucion_df = pd.DataFrame(resultado_ag['registro_evolucion'])
    evolucion_df = evolucion_df.drop(columns=['mejor_individuo'], errors='ignore')
    ruta_evolucion = os.path.join(RUTA_METRICAS, 'evolucion_ag.csv')
    evolucion_df.to_csv(ruta_evolucion, index=False, encoding='utf-8-sig')
    print(f"  [OK] Evolución AG: {ruta_evolucion}")
    
    # 5. Importancia de variables
    imp_base_df = pd.DataFrame(
        list(importancias_base.items()),
        columns=['Variable', 'Importancia']
    )
    ruta_imp = os.path.join(RUTA_METRICAS, 'importancia_variables_base.csv')
    imp_base_df.to_csv(ruta_imp, index=False, encoding='utf-8-sig')
    print(f"  [OK] Importancia variables: {ruta_imp}")
    
    imp_opt_df = pd.DataFrame(
        list(importancias_opt.items()),
        columns=['Variable', 'Importancia']
    )
    ruta_imp_opt = os.path.join(RUTA_METRICAS, 'importancia_variables_optimizado.csv')
    imp_opt_df.to_csv(ruta_imp_opt, index=False, encoding='utf-8-sig')
    print(f"  [OK] Importancia variables (opt): {ruta_imp_opt}")


if __name__ == '__main__':
    main()
