"""
Módulo de carga de datos.
Se encarga de leer los microdatos de la EPH desde el archivo TXT
y realizar las validaciones iniciales.
"""

import pandas as pd
import os
from codigo.configuracion import (
    ARCHIVO_PERSONAS, SEPARADOR_CSV, VARIABLE_ESTADO,
    VALOR_OCUPADO, VALOR_DESOCUPADO, NOMBRE_TARGET,
    NOMBRES_PREDICTORAS, VARIABLES_PREDICTORAS
)


def cargar_datos_personas(ruta_archivo=None):
    """
    Carga los microdatos individuales de la EPH desde el archivo TXT.

    Parámetros
    ----------
    ruta_archivo : str, opcional
        Ruta al archivo de personas. Si no se proporciona,
        usa la ruta definida en configuracion.py.

    Retorna
    -------
    pandas.DataFrame
        DataFrame con todos los registros del archivo de personas.
    """
    if ruta_archivo is None:
        ruta_archivo = ARCHIVO_PERSONAS

    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(
            f"No se encontró el archivo de datos: {ruta_archivo}\n"
            f"Asegúrese de colocar el archivo de microdatos de la EPH "
            f"en la carpeta 'datos/originales/'."
        )

    print(f"Cargando datos desde: {ruta_archivo}")
    datos = pd.read_csv(ruta_archivo, sep=SEPARADOR_CSV, low_memory=False)
    print(f"Datos cargados exitosamente: {datos.shape[0]} filas x {datos.shape[1]} columnas")

    return datos


def filtrar_pea(datos):
    """
    Filtra la Población Económicamente Activa (PEA).

    Según la documentación oficial del INDEC, la variable ESTADO indica:
        0 = Entrevista individual no realizada (se descarta)
        1 = Ocupado (se conserva)
        2 = Desocupado (se conserva)
        3 = Inactivo (se descarta)
        4 = Menor de 10 años (se descarta)

    Se conservan únicamente los registros con ESTADO = 1 (ocupado)
    y ESTADO = 2 (desocupado), que conforman la PEA.

    Parámetros
    ----------
    datos : pandas.DataFrame
        DataFrame completo con todos los registros.

    Retorna
    -------
    pandas.DataFrame
        DataFrame filtrado con solo la PEA.
    dict
        Diccionario con estadísticas del filtrado.
    """
    total_original = len(datos)

    # Distribución completa de ESTADO antes del filtro
    distribucion_estado = datos[VARIABLE_ESTADO].value_counts().sort_index()

    # Filtrar: conservar solo ocupados (1) y desocupados (2)
    datos_pea = datos[datos[VARIABLE_ESTADO].isin([VALOR_OCUPADO, VALOR_DESOCUPADO])].copy()

    total_pea = len(datos_pea)
    cantidad_ocupados = (datos_pea[VARIABLE_ESTADO] == VALOR_OCUPADO).sum()
    cantidad_desocupados = (datos_pea[VARIABLE_ESTADO] == VALOR_DESOCUPADO).sum()

    estadisticas = {
        'total_original': total_original,
        'total_pea': total_pea,
        'cantidad_ocupados': cantidad_ocupados,
        'cantidad_desocupados': cantidad_desocupados,
        'porcentaje_ocupados': cantidad_ocupados / total_pea * 100,
        'porcentaje_desocupados': cantidad_desocupados / total_pea * 100,
        'distribucion_estado_original': distribucion_estado,
        'registros_descartados': total_original - total_pea
    }

    print("\n=== FILTRADO DE LA PEA ===")
    print(f"Registros originales: {total_original:,}")
    print(f"Registros PEA (ESTADO 1 o 2): {total_pea:,}")
    print(f"Registros descartados: {estadisticas['registros_descartados']:,}")
    print(f"  - Ocupados (ESTADO=1): {cantidad_ocupados:,} ({estadisticas['porcentaje_ocupados']:.2f}%)")
    print(f"  - Desocupados (ESTADO=2): {cantidad_desocupados:,} ({estadisticas['porcentaje_desocupados']:.2f}%)")
    print(f"\nDistribución original de ESTADO:")
    for valor, conteo in distribucion_estado.items():
        etiquetas = {0: 'Entrevista no realizada', 1: 'Ocupado', 2: 'Desocupado',
                     3: 'Inactivo', 4: 'Menor de 10 años'}
        etiqueta = etiquetas.get(valor, f'Código {valor}')
        print(f"  {valor} ({etiqueta}): {conteo:,}")

    return datos_pea, estadisticas


def crear_target(datos_pea):
    """
    Crea la variable objetivo binaria 'desocupado'.

    Codificación:
        ESTADO = 1 (Ocupado)    → desocupado = 0
        ESTADO = 2 (Desocupado) → desocupado = 1

    Parámetros
    ----------
    datos_pea : pandas.DataFrame
        DataFrame filtrado con la PEA.

    Retorna
    -------
    pandas.DataFrame
        DataFrame con la nueva columna 'desocupado'.
    """
    datos_pea = datos_pea.copy()
    datos_pea[NOMBRE_TARGET] = (datos_pea[VARIABLE_ESTADO] == VALOR_DESOCUPADO).astype(int)

    print("\n=== VARIABLE OBJETIVO ===")
    print(f"Variable creada: '{NOMBRE_TARGET}'")
    print(f"  0 (Ocupado):    {(datos_pea[NOMBRE_TARGET] == 0).sum():,}")
    print(f"  1 (Desocupado): {(datos_pea[NOMBRE_TARGET] == 1).sum():,}")
    print(f"  Ratio desbalance: {(datos_pea[NOMBRE_TARGET] == 0).sum() / (datos_pea[NOMBRE_TARGET] == 1).sum():.1f}:1")

    return datos_pea


def mostrar_info_variables(datos_pea):
    """
    Muestra información detallada sobre las variables predictoras
    candidatas según la documentación oficial del INDEC.

    Parámetros
    ----------
    datos_pea : pandas.DataFrame
        DataFrame de la PEA.
    """
    print("\n=== VARIABLES PREDICTORAS CANDIDATAS ===")
    print(f"{'Variable':<12} {'Tipo':<12} {'Descripción':<50} {'Únicos':<8} {'Nulos':<8}")
    print("-" * 90)

    for nombre in NOMBRES_PREDICTORAS:
        if nombre in datos_pea.columns:
            info = VARIABLES_PREDICTORAS[nombre]
            n_unicos = datos_pea[nombre].nunique()
            n_nulos = datos_pea[nombre].isna().sum()
            print(f"{nombre:<12} {info['tipo']:<12} {info['descripcion']:<50} {n_unicos:<8} {n_nulos:<8}")

            # Mostrar distribución para variables categóricas
            if info['tipo'] in ['categorica', 'ordinal']:
                distribucion = datos_pea[nombre].value_counts().sort_index()
                for valor, conteo in distribucion.items():
                    etiqueta = info['codigos'].get(valor, '')
                    especial = ' [VALOR ESPECIAL]' if valor in info['valores_especiales'] else ''
                    print(f"  {valor:>6}: {conteo:>6,} ({conteo/len(datos_pea)*100:5.1f}%) {etiqueta}{especial}")
