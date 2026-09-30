"""
Paquete central de código para el Trabajo Práctico Integrador de Algoritmos Genéticos.

Contiene los módulos de configuración, carga de datos, preprocesamiento,
modelado base con Random Forest, optimización mediante Algoritmos Genéticos (DEAP),
evaluación de métricas y generación de visualizaciones.
"""

import sys

__version__ = "1.0.0"
__author__ = "Tomas Hidalgo"

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

