"""
Módulo de preprocesamiento de datos.
Realiza la limpieza, transformación y codificación de variables
para su uso en los modelos de Machine Learning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from codigo.configuracion import (
    NOMBRES_PREDICTORAS, VARIABLES_PREDICTORAS, NOMBRE_TARGET,
    VARIABLES_CATEGORICAS, VARIABLES_ORDINALES, VARIABLES_NUMERICAS,
    SEMILLA, PROPORCION_PRUEBA
)


def limpiar_datos(datos_pea):
    """
    Realiza la limpieza de las variables predictoras según la
    documentación oficial del INDEC.

    Tratamiento de valores especiales:
    - CH06 (edad): valor -1 indica menores de 1 año. En la PEA no deberían
      existir (son menores de 10 años). Si aparecen, se eliminan.
    - CH07 (estado civil): código 9 = Ns/Nr. Se eliminan por ser muy pocos.
    - CH08 (cobertura salud): código 9 = Ns/Nr. Se eliminan por ser pocos.
    - CH09 (asistencia escolar): código 9 = Ns/Nr. Se eliminan.
    - CH15 (lugar nacimiento): código 9 = Ns/Nr. Se eliminan por ser pocos.

    Parámetros
    ----------
    datos_pea : pandas.DataFrame
        DataFrame de la PEA con la variable objetivo ya creada.

    Retorna
    -------
    pandas.DataFrame
        DataFrame limpio listo para la codificación.
    dict
        Estadísticas de la limpieza realizada.
    """
    datos = datos_pea.copy()
    registros_inicial = len(datos)
    registros_eliminados = {}

    print("\n=== LIMPIEZA DE DATOS ===")

    # ----- CH06 (Edad) -----
    # Según la documentación del INDEC, -1 indica menores de 1 año.
    # En la PEA filtrada no deberían existir porque la PEA es para
    # personas de 10 años o más. Verificamos y eliminamos si existen.
    menores_invalidos = datos['CH06'] < 0
    if menores_invalidos.sum() > 0:
        print(f"CH06: Se encontraron {menores_invalidos.sum()} registros con edad < 0 (menores de 1 año).")
        print("  → Se eliminan porque no corresponden a la PEA.")
        registros_eliminados['CH06_negativos'] = menores_invalidos.sum()
        datos = datos[~menores_invalidos]
    else:
        print("CH06: No se encontraron valores negativos en la PEA. Correcto.")

    # ----- CH07 (Estado civil) -----
    # Código 9 = Ns/Nr
    ns_nr_ch07 = datos['CH07'] == 9
    if ns_nr_ch07.sum() > 0:
        print(f"CH07: Se encontraron {ns_nr_ch07.sum()} registros con código 9 (Ns/Nr).")
        porcentaje = ns_nr_ch07.sum() / len(datos) * 100
        print(f"  → Representan el {porcentaje:.3f}% de los datos.")
        print("  → Se eliminan por ser un porcentaje muy reducido y no aportar información.")
        registros_eliminados['CH07_ns_nr'] = ns_nr_ch07.sum()
        datos = datos[~ns_nr_ch07]
    else:
        print("CH07: No se encontraron valores Ns/Nr.")

    # ----- CH08 (Cobertura médica) -----
    # Código 9 = Ns/Nr
    ns_nr_ch08 = datos['CH08'] == 9
    if ns_nr_ch08.sum() > 0:
        print(f"CH08: Se encontraron {ns_nr_ch08.sum()} registros con código 9 (Ns/Nr).")
        porcentaje = ns_nr_ch08.sum() / len(datos) * 100
        print(f"  → Representan el {porcentaje:.3f}% de los datos.")
        print("  → Se eliminan por ser un porcentaje muy reducido.")
        registros_eliminados['CH08_ns_nr'] = ns_nr_ch08.sum()
        datos = datos[~ns_nr_ch08]
    else:
        print("CH08: No se encontraron valores Ns/Nr.")

    # ----- CH09 (Asistencia escolar) -----
    ns_nr_ch09 = datos['CH09'] == 9
    if ns_nr_ch09.sum() > 0:
        print(f"CH09: Se encontraron {ns_nr_ch09.sum()} registros con código 9 (Ns/Nr).")
        registros_eliminados['CH09_ns_nr'] = ns_nr_ch09.sum()
        datos = datos[~ns_nr_ch09]
    else:
        print("CH09: No se encontraron valores Ns/Nr.")

    # ----- CH15 (Lugar de nacimiento) -----
    ns_nr_ch15 = datos['CH15'] == 9
    if ns_nr_ch15.sum() > 0:
        print(f"CH15: Se encontraron {ns_nr_ch15.sum()} registros con código 9 (Ns/Nr).")
        porcentaje = ns_nr_ch15.sum() / len(datos) * 100
        print(f"  → Representan el {porcentaje:.3f}% de los datos.")
        print("  → Se eliminan por ser un porcentaje muy reducido.")
        registros_eliminados['CH15_ns_nr'] = ns_nr_ch15.sum()
        datos = datos[~ns_nr_ch15]
    else:
        print("CH15: No se encontraron valores Ns/Nr.")

    # ----- Verificación de valores faltantes (NaN) -----
    print("\nVerificación de valores faltantes (NaN) en variables predictoras:")
    for var in NOMBRES_PREDICTORAS:
        if var in datos.columns:
            n_nulos = datos[var].isna().sum()
            if n_nulos > 0:
                print(f"  {var}: {n_nulos} valores faltantes")
                # Eliminar registros con NaN en variables predictoras
                datos = datos.dropna(subset=[var])
                registros_eliminados[f'{var}_nulos'] = n_nulos
            else:
                print(f"  {var}: sin valores faltantes ✓")

    registros_final = len(datos)
    total_eliminados = registros_inicial - registros_final

    estadisticas_limpieza = {
        'registros_inicial': registros_inicial,
        'registros_final': registros_final,
        'total_eliminados': total_eliminados,
        'detalle_eliminados': registros_eliminados,
        'porcentaje_conservado': registros_final / registros_inicial * 100
    }

    print(f"\nResumen de limpieza:")
    print(f"  Registros iniciales: {registros_inicial:,}")
    print(f"  Registros eliminados: {total_eliminados:,}")
    print(f"  Registros finales: {registros_final:,}")
    print(f"  Porcentaje conservado: {estadisticas_limpieza['porcentaje_conservado']:.2f}%")

    return datos, estadisticas_limpieza


def preparar_predictores(datos):
    """
    Selecciona y prepara las variables predictoras.

    NIVEL_ED se trata como ordinal por su jerarquía natural
    (sin instrucción < primaria < secundaria < superior).

    Parámetros
    ----------
    datos : pandas.DataFrame

    Retorna
    -------
    pandas.DataFrame, pandas.Series
    """
    # Seleccionar variables predictoras disponibles
    variables_disponibles = [v for v in NOMBRES_PREDICTORAS if v in datos.columns]
    print(f"\nVariables predictoras seleccionadas: {len(variables_disponibles)}")
    for var in variables_disponibles:
        info = VARIABLES_PREDICTORAS[var]
        print(f"  - {var}: {info['descripcion']} ({info['tipo']})")

    predictores = datos[variables_disponibles].copy()
    objetivo = datos[NOMBRE_TARGET].copy()

    # Codificar NIVEL_ED como ordinal
    if 'NIVEL_ED' in predictores.columns:
        orden = VARIABLES_PREDICTORAS['NIVEL_ED']['orden_ordinal']
        predictores['NIVEL_ED'] = predictores['NIVEL_ED'].map(orden)
        print("\nNIVEL_ED codificado como ordinal:")
        print("  7 (Sin instrucción) → 0")
        print("  1 (Primaria incompleta) → 1")
        print("  2 (Primaria completa) → 2")
        print("  3 (Secundaria incompleta) → 3")
        print("  4 (Secundaria completa) → 4")
        print("  5 (Superior universitaria incompleta) → 5")
        print("  6 (Superior universitaria completa) → 6")

    return predictores, objetivo


def crear_pipeline_codificacion(predictores):
    """
    Crea el ColumnTransformer para codificación.
    OHE para categóricas; numéricas y ordinales pasan sin modificar.
    El encoder se ajusta solo con datos de entrenamiento (sin leakage).

    Parámetros
    ----------
    predictores : pandas.DataFrame

    Retorna
    -------
    sklearn.compose.ColumnTransformer
    """
    columnas_categoricas = [c for c in VARIABLES_CATEGORICAS if c in predictores.columns]
    columnas_numericas = [c for c in VARIABLES_NUMERICAS if c in predictores.columns]
    columnas_ordinales = [c for c in VARIABLES_ORDINALES if c in predictores.columns]

    # Columnas que pasan sin transformar
    columnas_passthrough = columnas_numericas + columnas_ordinales

    transformadores = []

    if columnas_categoricas:
        transformadores.append(
            ('categoricas', OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore',
                drop=None  # No eliminar ninguna categoría para preservar interpretabilidad
            ), columnas_categoricas)
        )

    if columnas_passthrough:
        transformadores.append(
            ('passthrough', 'passthrough', columnas_passthrough)
        )

    codificador = ColumnTransformer(
        transformers=transformadores,
        remainder='drop'
    )

    print(f"\nPipeline de codificación creado:")
    print(f"  Variables categóricas (One-Hot): {columnas_categoricas}")
    print(f"  Variables numéricas (sin cambio): {columnas_numericas}")
    print(f"  Variables ordinales (ya codificadas): {columnas_ordinales}")

    return codificador


def dividir_datos(predictores, objetivo):
    """
    Divide en train/test con estratificación para mantener
    la proporción de clases (importante con desbalance ~94/6%).

    Parámetros
    ----------
    predictores : pandas.DataFrame
    objetivo : pandas.Series

    Retorna
    -------
    tuple : (X_entrenamiento, X_prueba, y_entrenamiento, y_prueba)
    """
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        predictores,
        objetivo,
        test_size=PROPORCION_PRUEBA,
        stratify=objetivo,
        random_state=SEMILLA
    )

    print("\n=== DIVISIÓN DE DATOS ===")
    print(f"Proporción prueba: {PROPORCION_PRUEBA*100:.0f}%")
    print(f"Estratificado: Sí (preservando proporción de clases)")
    print(f"Semilla: {SEMILLA}")
    print(f"\nConjunto de entrenamiento: {len(X_entrenamiento):,} registros")
    print(f"  Ocupados: {(y_entrenamiento == 0).sum():,} ({(y_entrenamiento == 0).sum()/len(y_entrenamiento)*100:.2f}%)")
    print(f"  Desocupados: {(y_entrenamiento == 1).sum():,} ({(y_entrenamiento == 1).sum()/len(y_entrenamiento)*100:.2f}%)")
    print(f"\nConjunto de prueba: {len(X_prueba):,} registros")
    print(f"  Ocupados: {(y_prueba == 0).sum():,} ({(y_prueba == 0).sum()/len(y_prueba)*100:.2f}%)")
    print(f"  Desocupados: {(y_prueba == 1).sum():,} ({(y_prueba == 1).sum()/len(y_prueba)*100:.2f}%)")

    print("\n  → La división estratificada garantiza que ambos conjuntos")
    print("    mantengan la misma proporción de clases, lo cual es crítico")
    print("    cuando hay desbalance significativo.")

    return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba


def ejecutar_preprocesamiento_completo(datos_pea):
    """
    Ejecuta todo el flujo de preprocesamiento de datos.

    Parámetros
    ----------
    datos_pea : pandas.DataFrame
        DataFrame de la PEA con la variable objetivo.

    Retorna
    -------
    dict
        Diccionario con todos los datos preprocesados y metadatos.
    """
    # 1. Limpiar datos
    datos_limpios, estadisticas_limpieza = limpiar_datos(datos_pea)

    # 2. Preparar predictores
    predictores, objetivo = preparar_predictores(datos_limpios)

    # 3. Dividir datos
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = dividir_datos(predictores, objetivo)

    # 4. Crear pipeline de codificación
    codificador = crear_pipeline_codificacion(X_entrenamiento)

    # 5. Ajustar y transformar SOLO con datos de entrenamiento
    X_entrenamiento_cod = codificador.fit_transform(X_entrenamiento)
    X_prueba_cod = codificador.transform(X_prueba)

    # 6. Obtener nombres de columnas resultantes
    nombres_columnas = obtener_nombres_columnas(codificador, X_entrenamiento)

    print(f"\n=== CODIFICACIÓN COMPLETADA ===")
    print(f"Variables originales: {X_entrenamiento.shape[1]}")
    print(f"Variables después de One-Hot Encoding: {X_entrenamiento_cod.shape[1]}")
    print(f"  → El codificador se ajustó SOLO con datos de entrenamiento")
    print(f"  → Los datos de prueba se transformaron con el mismo codificador")
    print(f"  → No existe data leakage")

    return {
        'X_entrenamiento': X_entrenamiento_cod,
        'X_prueba': X_prueba_cod,
        'y_entrenamiento': y_entrenamiento,
        'y_prueba': y_prueba,
        'codificador': codificador,
        'nombres_columnas': nombres_columnas,
        'estadisticas_limpieza': estadisticas_limpieza,
        'predictores_originales': predictores,
        'datos_limpios': datos_limpios
    }


def obtener_nombres_columnas(codificador, X_referencia):
    """
    Obtiene los nombres de las columnas resultantes después de la codificación.

    Parámetros
    ----------
    codificador : ColumnTransformer
        El transformador ya ajustado.
    X_referencia : pandas.DataFrame
        DataFrame de referencia para los nombres.

    Retorna
    -------
    list
        Lista de nombres de columnas codificadas.
    """
    nombres = []

    for nombre_transf, transformador, columnas in codificador.transformers_:
        if nombre_transf == 'categoricas' and hasattr(transformador, 'get_feature_names_out'):
            nombres_ohe = transformador.get_feature_names_out(columnas)
            nombres.extend(nombres_ohe)
        elif nombre_transf == 'passthrough':
            nombres.extend(columnas)
        elif nombre_transf == 'remainder':
            pass  # Se ignoraron

    return nombres
