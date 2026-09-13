"""
Configuración central del proyecto.
Contiene rutas, parámetros del modelo, del Algoritmo Genético
y constantes generales utilizadas en todo el proyecto.
"""

import os

# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

# Directorio raíz del proyecto
DIRECTORIO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Rutas de datos
RUTA_DATOS_ORIGINALES = os.path.join(DIRECTORIO_RAIZ, 'datos', 'originales')
RUTA_DATOS_PROCESADOS = os.path.join(DIRECTORIO_RAIZ, 'datos', 'procesados')

# Archivo de microdatos individuales de la EPH
ARCHIVO_PERSONAS = os.path.join(RUTA_DATOS_ORIGINALES, 'personas_tot.urb_3T_2025.txt')

# Rutas de resultados
RUTA_GRAFICOS = os.path.join(DIRECTORIO_RAIZ, 'resultados', 'graficos')
RUTA_METRICAS = os.path.join(DIRECTORIO_RAIZ, 'resultados', 'metricas')
RUTA_MODELOS = os.path.join(DIRECTORIO_RAIZ, 'resultados', 'modelos')

# ============================================================
# SEMILLA DE REPRODUCIBILIDAD
# ============================================================

SEMILLA = 42

# ============================================================
# INFORMACIÓN DE LA FUENTE DE DATOS
# ============================================================

FUENTE_DATOS = "INDEC - Encuesta Permanente de Hogares (EPH) - Total Urbano"
PERIODO_DATOS = "Tercer trimestre 2025"
SEPARADOR_CSV = ';'

# ============================================================
# DEFINICIÓN DE VARIABLES SEGÚN DISEÑO DE REGISTRO INDEC
# ============================================================

# Variable objetivo
# ESTADO: Condición de actividad
#   0 = Entrevista individual no realizada
#   1 = Ocupado
#   2 = Desocupado
#   3 = Inactivo
#   4 = Menor de 10 años
VARIABLE_ESTADO = 'ESTADO'
VALOR_OCUPADO = 1
VALOR_DESOCUPADO = 2
NOMBRE_TARGET = 'desocupado'

# Variables predictoras candidatas (según diseño de registro EPH 3T 2025)
# Se seleccionan variables sociodemográficas relevantes para el problema

VARIABLES_PREDICTORAS = {
    # --- Variables demográficas básicas ---
    'CH04': {
        'descripcion': 'Sexo',
        'tipo': 'categorica',
        'codigos': {1: 'Varón', 2: 'Mujer'},
        'valores_especiales': []
    },
    'CH06': {
        'descripcion': 'Edad (años cumplidos)',
        'tipo': 'numerica',
        'codigos': {},
        'valores_especiales': [-1]  # -1 = menor de 1 año
    },
    'CH07': {
        'descripcion': 'Estado civil',
        'tipo': 'categorica',
        'codigos': {
            1: 'Unido/a',
            2: 'Casado/a',
            3: 'Separado/a o divorciado/a',
            4: 'Viudo/a',
            5: 'Soltero/a'
        },
        'valores_especiales': [9]  # 9 = Ns/Nr
    },
    'CH08': {
        'descripcion': 'Cobertura médica',
        'tipo': 'categorica',
        'codigos': {
            1: 'Obra social (incluye PAMI)',
            2: 'Mutual/prepaga/servicio de emergencia',
            3: 'Planes y seguros públicos',
            4: 'No paga ni le descuentan',
            12: 'Obra social y mutual/prepaga',
            13: 'Obra social y planes/seguros públicos',
            23: 'Mutual/prepaga y planes/seguros públicos',
            123: 'Obra social, mutual/prepaga y planes/seguros públicos'
        },
        'valores_especiales': [9]  # 9 = Ns/Nr
    },
    'CH03': {
        'descripcion': 'Relación de parentesco con el jefe/a del hogar',
        'tipo': 'categorica',
        'codigos': {
            1: 'Jefe/a',
            2: 'Cónyuge/pareja',
            3: 'Hijo/a / hijastro/a',
            4: 'Yerno/nuera',
            5: 'Nieto/a',
            6: 'Madre/padre/suegro/a',
            7: 'Otros familiares',
            8: 'No familiares',
            9: 'Miembros del hogar de servicio doméstico',
            10: 'Miembros del hogar de servicio doméstico (familiares)'
        },
        'valores_especiales': []
    },
    'NIVEL_ED': {
        'descripcion': 'Nivel educativo',
        'tipo': 'ordinal',
        'codigos': {
            1: 'Primaria incompleta (incluye educación especial)',
            2: 'Primaria completa',
            3: 'Secundaria incompleta',
            4: 'Secundaria completa',
            5: 'Superior universitaria incompleta',
            6: 'Superior universitaria completa',
            7: 'Sin instrucción'
        },
        'valores_especiales': [],
        # Orden para tratamiento ordinal (de menor a mayor nivel)
        'orden_ordinal': {7: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    },
    'CH09': {
        'descripcion': 'Asistencia a establecimiento educativo',
        'tipo': 'categorica',
        'codigos': {1: 'Sí, asiste', 2: 'No asiste pero asistió', 3: 'Nunca asistió'},
        'valores_especiales': [9]  # 9 = Ns/Nr
    },
    'CH15': {
        'descripcion': 'Lugar de nacimiento',
        'tipo': 'categorica',
        'codigos': {
            1: 'En esta localidad',
            2: 'En otra localidad de esta provincia',
            3: 'En otra provincia',
            4: 'En un país limítrofe',
            5: 'En otro país'
        },
        'valores_especiales': [9]  # 9 = Ns/Nr
    },
    'AGLOMERADO': {
        'descripcion': 'Aglomerado urbano',
        'tipo': 'categorica',
        'codigos': {},  # 54 aglomerados, se codifica con one-hot
        'valores_especiales': []
    },
}

# Lista de nombres de variables predictoras
NOMBRES_PREDICTORAS = list(VARIABLES_PREDICTORAS.keys())

# Variables categóricas para One-Hot Encoding (sin relación ordinal)
VARIABLES_CATEGORICAS = [
    nombre for nombre, info in VARIABLES_PREDICTORAS.items()
    if info['tipo'] == 'categorica'
]

# Variables ordinales
VARIABLES_ORDINALES = [
    nombre for nombre, info in VARIABLES_PREDICTORAS.items()
    if info['tipo'] == 'ordinal'
]

# Variables numéricas
VARIABLES_NUMERICAS = [
    nombre for nombre, info in VARIABLES_PREDICTORAS.items()
    if info['tipo'] == 'numerica'
]

# ============================================================
# PARÁMETROS DE DIVISIÓN DE DATOS
# ============================================================

PROPORCION_PRUEBA = 0.20  # 20% para prueba
ESTRATIFICADO = True

# ============================================================
# PARÁMETROS DEL MODELO BASE (Random Forest)
# ============================================================

PARAMETROS_RF = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'class_weight': 'balanced',  # Compensar desbalance de clases
    'random_state': SEMILLA,
    'n_jobs': -1
}

# ============================================================
# PARÁMETROS DEL ALGORITMO GENÉTICO
# ============================================================

PARAMETROS_AG = {
    'tamano_poblacion': 30,
    'numero_generaciones': 30,
    'probabilidad_cruce': 0.8,
    'probabilidad_mutacion': 0.1,
    'tamano_torneo': 3,
    'numero_elite': 2,  # Individuos que pasan directamente
    'penalizacion_variables': 0.005,  # Penalización por cada variable
}

# Parámetros de validación cruzada dentro del AG
FOLDS_CV = 5

# ============================================================
# PARÁMETROS DE VISUALIZACIÓN
# ============================================================

COLORES = {
    'ocupado': '#2ecc71',      # Verde
    'desocupado': '#e74c3c',   # Rojo
    'modelo_base': '#3498db',  # Azul
    'modelo_opt': '#e67e22',   # Naranja
    'fitness': '#9b59b6',      # Púrpura
    'fondo': '#f8f9fa',        # Gris claro
}

TAMANO_FIGURA = (10, 6)
DPI_GRAFICOS = 150
FORMATO_GRAFICOS = 'png'
