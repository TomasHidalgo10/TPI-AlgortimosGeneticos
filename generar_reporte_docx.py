"""
=============================================================================
Generador de Reporte DOCX de Resultados.

Lee los resultados generados por el proyecto (JSON, CSVs, graficos PNG)
y genera un documento Word (.docx) con analisis completo, tablas y graficas.

Uso:
    python generar_reporte_docx.py
=============================================================================
"""

import json
import os
import sys
import csv
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

# Directorio raiz del proyecto
DIRECTORIO_RAIZ = os.path.dirname(os.path.abspath(__file__))
RUTA_METRICAS = os.path.join(DIRECTORIO_RAIZ, 'resultados', 'metricas')
RUTA_GRAFICOS = os.path.join(DIRECTORIO_RAIZ, 'resultados', 'graficos')
RUTA_SALIDA = os.path.join(DIRECTORIO_RAIZ, 'resultados')


# ============================================================
# FUNCIONES DE CARGA DE DATOS
# ============================================================

def cargar_json(ruta: str) -> Dict[str, Any]:
    with open(ruta, 'r', encoding='utf-8') as f:
        return json.load(f)


def cargar_csv(ruta: str) -> List[Dict[str, str]]:
    with open(ruta, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader)



def cargar_todos_los_datos():
    datos = {}

    ruta_json = os.path.join(RUTA_METRICAS, 'resultados_completos.json')
    if os.path.exists(ruta_json):
        datos['resultados'] = cargar_json(ruta_json)
        print("  OK Cargado: resultados_completos.json")
    else:
        print(f"  ERROR No se encontro: {ruta_json}")
        sys.exit(1)

    ruta_comp = os.path.join(RUTA_METRICAS, 'comparacion_modelos.csv')
    if os.path.exists(ruta_comp):
        datos['comparacion'] = cargar_csv(ruta_comp)
        print("  OK Cargado: comparacion_modelos.csv")

    ruta_vars = os.path.join(RUTA_METRICAS, 'variables_seleccionadas.csv')
    if os.path.exists(ruta_vars):
        datos['variables'] = cargar_csv(ruta_vars)
        print("  OK Cargado: variables_seleccionadas.csv")

    ruta_evol = os.path.join(RUTA_METRICAS, 'evolucion_ag.csv')
    if os.path.exists(ruta_evol):
        datos['evolucion'] = cargar_csv(ruta_evol)
        print("  OK Cargado: evolucion_ag.csv")

    ruta_imp = os.path.join(RUTA_METRICAS, 'importancia_variables_base.csv')
    if os.path.exists(ruta_imp):
        datos['importancia_base'] = cargar_csv(ruta_imp)
        print("  OK Cargado: importancia_variables_base.csv")

    ruta_imp_opt = os.path.join(RUTA_METRICAS, 'importancia_variables_optimizado.csv')
    if os.path.exists(ruta_imp_opt):
        datos['importancia_opt'] = cargar_csv(ruta_imp_opt)
        print("  OK Cargado: importancia_variables_optimizado.csv")

    datos['graficos'] = {}
    archivos_graficos = [
        ('distribucion_clases', '01_distribucion_clases.png'),
        ('evolucion_fitness', '02_evolucion_fitness.png'),
        ('evolucion_variables', '03_evolucion_variables.png'),
        ('comparacion_metricas', '04_comparacion_metricas.png'),
        ('matriz_base', '05_matriz_confusion_base.png'),
        ('matriz_opt', '06_matriz_confusion_optimizado.png'),
        ('importancia_vars', '07_importancia_variables.png'),
        ('exploratorio', '08_analisis_exploratorio.png'),
    ]
    for clave, archivo in archivos_graficos:
        ruta = os.path.join(RUTA_GRAFICOS, archivo)
        if os.path.exists(ruta):
            datos['graficos'][clave] = ruta
            print(f"  OK Grafico encontrado: {archivo}")
        else:
            print(f"  AVISO Grafico no encontrado: {archivo}")

    return datos


# ============================================================
# FUNCIONES AUXILIARES PARA EL DOCUMENTO
# ============================================================

def set_cell_shading(cell, color_hex):
    """Aplica color de fondo a una celda."""
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def formato_celda(cell, texto, bold=False, size=9, alignment=WD_ALIGN_PARAGRAPH.LEFT,
                  color=None, font_name='Calibri'):
    """Formatea una celda de tabla."""
    cell.text = ''
    p = cell.paragraphs[0]
    p.alignment = alignment
    run = p.add_run(str(texto))
    run.font.size = Pt(size)
    run.font.name = font_name
    run.bold = bold
    if color:
        run.font.color.rgb = color
    # Reducir espaciado del parrafo
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)


def agregar_tabla_estilizada(doc, encabezados, filas, col_widths=None):
    """Crea una tabla con estilo profesional."""
    tabla = doc.add_table(rows=1 + len(filas), cols=len(encabezados))
    tabla.alignment = WD_TABLE_ALIGNMENT.CENTER
    tabla.style = 'Table Grid'

    # Encabezados
    header_row = tabla.rows[0]
    for i, enc in enumerate(encabezados):
        cell = header_row.cells[i]
        formato_celda(cell, enc, bold=True, size=9,
                      alignment=WD_ALIGN_PARAGRAPH.CENTER,
                      color=RGBColor(0xFF, 0xFF, 0xFF))
        set_cell_shading(cell, '2B579A')

    # Filas de datos
    for r, fila in enumerate(filas):
        row = tabla.rows[r + 1]
        for c, valor in enumerate(fila):
            cell = row.cells[c]
            alineacion = WD_ALIGN_PARAGRAPH.LEFT if c == 0 else WD_ALIGN_PARAGRAPH.CENTER
            formato_celda(cell, str(valor), size=9, alignment=alineacion)
            # Alternar color de fondo
            if r % 2 == 0:
                set_cell_shading(cell, 'F2F2F2')

    # Aplicar anchos de columna si se proporcionan
    if col_widths:
        for i, width in enumerate(col_widths):
            for row in tabla.rows:
                row.cells[i].width = Cm(width)

    return tabla


def agregar_grafico(doc, ruta_grafico, ancho=Inches(5.8), titulo=None):
    """Agrega un grafico al documento."""
    if titulo:
        p = doc.add_paragraph()
        run = p.add_run(titulo)
        run.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(4)

    if ruta_grafico and os.path.exists(ruta_grafico):
        doc.add_picture(ruta_grafico, width=ancho)
        # Centrar la imagen
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        p = doc.add_paragraph('[Grafico no disponible]')
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].font.color.rgb = RGBColor(0x99, 0x99, 0x99)


def agregar_parrafo(doc, texto, size=10.5, bold=False, italic=False,
                     color=None, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
                     space_after=Pt(6), space_before=Pt(0)):
    """Agrega un parrafo formateado."""
    p = doc.add_paragraph()
    run = p.add_run(texto)
    run.font.size = Pt(size)
    run.font.name = 'Calibri'
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = color
    p.alignment = alignment
    p.paragraph_format.space_after = space_after
    p.paragraph_format.space_before = space_before
    return p


def agregar_bullet(doc, texto, nivel=0, size=10, bold_prefix=None):
    """Agrega un item con vineta."""
    p = doc.add_paragraph(style='List Bullet')
    if bold_prefix:
        run_bold = p.add_run(bold_prefix)
        run_bold.bold = True
        run_bold.font.size = Pt(size)
        run_bold.font.name = 'Calibri'
        run_rest = p.add_run(texto)
        run_rest.font.size = Pt(size)
        run_rest.font.name = 'Calibri'
    else:
        p.text = ''
        run = p.add_run(texto)
        run.font.size = Pt(size)
        run.font.name = 'Calibri'
    p.paragraph_format.space_after = Pt(3)
    return p


# ============================================================
# GENERACION DEL DOCUMENTO
# ============================================================

def generar_documento(datos):
    """Genera el documento DOCX completo."""

    doc = Document()

    res = datos['resultados']
    base = res['modelo_base']
    opt = res['modelo_optimizado']
    ag = res['algoritmo_genetico']
    info_datos = res.get('datos', {})

    # Metricas derivadas
    reduccion_vars = (1 - ag['n_variables_seleccionadas'] / ag['n_variables_total']) * 100
    diff_accuracy = opt['accuracy'] - base['accuracy']
    diff_f1 = opt['f1_desocupado'] - base['f1_desocupado']
    diff_recall = opt['recall_desocupado'] - base['recall_desocupado']
    diff_precision = opt['precision_desocupado'] - base['precision_desocupado']
    diff_f1_macro = opt['f1_macro'] - base['f1_macro']
    tiempo_evol_min = ag['tiempo_evolucion'] / 60
    total_pea = info_datos.get('total_pea', 0)
    ocupados = info_datos.get('ocupados', 0)
    desocupados = info_datos.get('desocupados', 0)
    tasa_desoc = (desocupados / total_pea * 100) if total_pea > 0 else 0
    params = ag.get('parametros', {})

    mc_base = base.get('matriz_confusion', [[0, 0], [0, 0]])
    mc_opt = opt.get('matriz_confusion', [[0, 0], [0, 0]])

    # ============================================================
    # CONFIGURAR ESTILOS DEL DOCUMENTO
    # ============================================================
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(10.5)
    style.paragraph_format.line_spacing = 1.15

    # Configurar margenes
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    # ============================================================
    # PORTADA
    # ============================================================
    # Espaciado superior
    for _ in range(4):
        doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('REPORTE DE RESULTADOS')
    run.font.size = Pt(28)
    run.font.color.rgb = RGBColor(0x2B, 0x57, 0x9A)
    run.bold = True
    run.font.name = 'Calibri'

    doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('Prediccion de la Condicion de Desocupacion\n'
                     'en la Poblacion Economicamente Activa de Argentina')
    run.font.size = Pt(16)
    run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    run.font.name = 'Calibri'

    doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('Machine Learning con seleccion de variables\n'
                     'optimizada por Algoritmos Geneticos')
    run.font.size = Pt(13)
    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    run.italic = True
    run.font.name = 'Calibri'

    for _ in range(3):
        doc.add_paragraph()

    # Info de portada
    info_portada = [
        ('Fuente de datos', 'INDEC - EPH Total Urbano - 3er Trimestre 2025'),
        ('Modelo utilizado', 'Random Forest (scikit-learn)'),
        ('Optimizacion', 'Algoritmo Genetico (DEAP)'),
        ('Fecha del reporte', datetime.now().strftime('%d/%m/%Y')),
    ]
    tabla_portada = doc.add_table(rows=len(info_portada), cols=2)
    tabla_portada.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (clave, valor) in enumerate(info_portada):
        cell_k = tabla_portada.rows[i].cells[0]
        cell_v = tabla_portada.rows[i].cells[1]
        formato_celda(cell_k, clave, bold=True, size=10,
                      alignment=WD_ALIGN_PARAGRAPH.RIGHT,
                      color=RGBColor(0x2B, 0x57, 0x9A))
        formato_celda(cell_v, valor, size=10)
        cell_k.width = Cm(5)
        cell_v.width = Cm(10)

    # Salto de pagina
    doc.add_page_break()

    # ============================================================
    # INDICE
    # ============================================================
    doc.add_heading('Indice', level=1)
    secciones_indice = [
        '1. Resumen Ejecutivo',
        '2. Descripcion del Dataset',
        '3. Analisis Exploratorio',
        '4. Modelo Base (Random Forest)',
        '5. Algoritmo Genetico',
        '6. Modelo Optimizado',
        '7. Comparacion de Modelos',
        '8. Seleccion de Variables',
        '9. Conclusiones',
    ]
    for sec in secciones_indice:
        p = doc.add_paragraph(sec)
        p.paragraph_format.space_after = Pt(2)
        p.runs[0].font.size = Pt(10.5)

    doc.add_page_break()

    # ============================================================
    # 1. RESUMEN EJECUTIVO
    # ============================================================
    doc.add_heading('1. Resumen Ejecutivo', level=1)

    agregar_parrafo(doc,
        'Este documento presenta los resultados obtenidos en el proyecto de '
        'prediccion de la condicion de desocupacion en la Poblacion Economicamente '
        'Activa (PEA) de Argentina, utilizando datos de la Encuesta Permanente de '
        'Hogares (EPH) del INDEC, correspondientes al 3er trimestre de 2025.'
    )
    agregar_parrafo(doc,
        'El objetivo del proyecto fue evaluar la capacidad de un Algoritmo Genetico '
        'para seleccionar un subconjunto optimo de variables predictoras que permita '
        'mantener o mejorar el desempeno de un modelo Random Forest en la tarea de '
        'clasificacion binaria (ocupado vs. desocupado).'
    )

    doc.add_heading('Indicadores Clave', level=2)

    indicadores = [
        ['Indicador', 'Modelo Base', 'Modelo Optimizado', 'Diferencia'],
        ['Variables utilizadas', str(base['n_variables']),
         str(opt['n_variables']),
         f"-{int(base['n_variables'] - opt['n_variables'])} ({reduccion_vars:.1f}%)"],
        ['Accuracy', f"{base['accuracy']:.4f}", f"{opt['accuracy']:.4f}",
         f"{diff_accuracy:+.4f}"],
        ['F1-Score (desocupado)', f"{base['f1_desocupado']:.4f}",
         f"{opt['f1_desocupado']:.4f}", f"{diff_f1:+.6f}"],
        ['Recall (desocupado)', f"{base['recall_desocupado']:.4f}",
         f"{opt['recall_desocupado']:.4f}", f"{diff_recall:+.4f}"],
        ['Precision (desocupado)', f"{base['precision_desocupado']:.4f}",
         f"{opt['precision_desocupado']:.4f}", f"{diff_precision:+.4f}"],
        ['F1 Macro', f"{base['f1_macro']:.4f}", f"{opt['f1_macro']:.4f}",
         f"{diff_f1_macro:+.4f}"],
        ['Mejor Fitness AG', '-', f"{ag['mejor_fitness']:.4f}", '-'],
        ['Tiempo evolucion AG', '-', f"{tiempo_evol_min:.1f} min", '-'],
    ]
    agregar_tabla_estilizada(doc, indicadores[0], indicadores[1:],
                              col_widths=[5, 3.5, 3.5, 3.5])

    doc.add_paragraph()

    # Veredicto
    if diff_f1 > 0.01:
        veredicto = 'MEJORO el F1 de desocupados'
    elif diff_f1 > -0.01:
        veredicto = 'MANTUVO el F1 de desocupados (diferencia despreciable)'
    else:
        veredicto = 'REDUJO el F1 de desocupados'

    agregar_parrafo(doc,
        f'Resultado principal: El Algoritmo Genetico {veredicto}, '
        f'reduciendo la dimensionalidad en un {reduccion_vars:.1f}% '
        f'(de {ag["n_variables_total"]} a {ag["n_variables_seleccionadas"]} variables).',
        bold=True, size=10.5,
        color=RGBColor(0x2B, 0x57, 0x9A)
    )

    doc.add_page_break()

    # ============================================================
    # 2. DESCRIPCION DEL DATASET
    # ============================================================
    doc.add_heading('2. Descripcion del Dataset', level=1)

    agregar_parrafo(doc,
        'Los datos provienen de la Encuesta Permanente de Hogares (EPH) del INDEC, '
        'correspondientes al relevamiento de Total Urbano del 3er trimestre de 2025. '
        'La EPH es la principal encuesta sociodemografica de Argentina y releva '
        'informacion sobre la situacion laboral de la poblacion.'
    )

    doc.add_heading('Composicion del dataset', level=2)

    datos_dataset = [
        ['Concepto', 'Cantidad', 'Observacion'],
        ['Registros originales (EPH)', f"{info_datos.get('total_original', 0):,}", 'Todas las personas encuestadas'],
        ['PEA (filtrada)', f"{total_pea:,}", 'Ocupados + Desocupados (ESTADO = 1 o 2)'],
        ['Ocupados', f"{ocupados:,}", f"{(ocupados/total_pea*100) if total_pea > 0 else 0:.1f}% de la PEA"],
        ['Desocupados', f"{desocupados:,}", f"{tasa_desoc:.1f}% de la PEA (clase minoritaria)"],
        ['Division entrenamiento', '80%', 'Estratificada por clase'],
        ['Division prueba', '20%', 'Estratificada por clase'],
    ]
    agregar_tabla_estilizada(doc, datos_dataset[0], datos_dataset[1:],
                              col_widths=[4.5, 3, 8])

    doc.add_paragraph()
    agregar_parrafo(doc,
        f'El dataset presenta un fuerte desbalance de clases: solo el {tasa_desoc:.1f}% '
        f'de los registros corresponden a personas desocupadas. Este desbalance fue '
        f'abordado mediante el uso de class_weight="balanced" en el modelo Random Forest, '
        f'que asigna pesos inversamente proporcionales a la frecuencia de cada clase.',
        italic=True
    )

    doc.add_heading('Variables predictoras', level=2)

    agregar_parrafo(doc,
        'Se utilizaron 9 variables originales del diseno de registro de la EPH, '
        'seleccionadas por su relevancia sociodemografica para el problema de '
        'prediccion de desocupacion:'
    )

    variables_info = [
        ['Variable', 'Descripcion', 'Tipo', 'Categorias codificadas'],
        ['CH04', 'Sexo', 'Categorica', '2'],
        ['CH06', 'Edad (anios cumplidos)', 'Numerica', '1 (sin codificar)'],
        ['CH07', 'Estado civil', 'Categorica', '5'],
        ['CH08', 'Cobertura medica', 'Categorica', '7'],
        ['CH03', 'Relacion de parentesco', 'Categorica', '10'],
        ['CH09', 'Asistencia educativa', 'Categorica', '2'],
        ['CH15', 'Lugar de nacimiento', 'Categorica', '5'],
        ['NIVEL_ED', 'Nivel educativo', 'Ordinal', '1 (mapeado)'],
        ['AGLOMERADO', 'Aglomerado urbano', 'Categorica', '54'],
    ]
    agregar_tabla_estilizada(doc, variables_info[0], variables_info[1:],
                              col_widths=[2.5, 4.5, 2.5, 3.5])

    doc.add_paragraph()
    agregar_parrafo(doc,
        f'Tras la codificacion One-Hot de las variables categoricas y el mapeo ordinal, '
        f'el dataset quedo con {ag["n_variables_total"]} variables predictoras en total.'
    )

    doc.add_page_break()

    # ============================================================
    # 3. ANALISIS EXPLORATORIO
    # ============================================================
    doc.add_heading('3. Analisis Exploratorio', level=1)

    agregar_parrafo(doc,
        'A continuacion se presentan visualizaciones que permiten comprender la '
        'distribucion de la variable objetivo y su relacion con las principales '
        'variables sociodemograficas.'
    )

    doc.add_heading('Distribucion de la variable objetivo', level=2)
    agregar_parrafo(doc,
        'La variable objetivo es binaria: 0 = Ocupado, 1 = Desocupado. '
        f'El grafico muestra el marcado desbalance entre clases, con solo {desocupados:,} '
        f'desocupados frente a {ocupados:,} ocupados.',
        size=10
    )
    agregar_grafico(doc, datos['graficos'].get('distribucion_clases'),
                     titulo='Figura 1: Distribucion de clases en la PEA')

    doc.add_paragraph()

    doc.add_heading('Relacion con variables sociodemograficas', level=2)
    agregar_parrafo(doc,
        'El analisis exploratorio revela patrones relevantes: la tasa de desocupacion '
        'varia significativamente segun sexo, edad, nivel educativo y estado civil. '
        'Los jovenes y las personas con menor nivel educativo presentan tasas de '
        'desocupacion mas altas.',
        size=10
    )
    agregar_grafico(doc, datos['graficos'].get('exploratorio'),
                     titulo='Figura 2: Analisis exploratorio de variables sociodemograficas')

    doc.add_page_break()

    # ============================================================
    # 4. MODELO BASE
    # ============================================================
    doc.add_heading('4. Modelo Base (Random Forest)', level=1)

    agregar_parrafo(doc,
        'Se entreno un modelo Random Forest como linea base, utilizando todas las '
        f'{base["n_variables"]} variables codificadas. Los parametros del modelo fueron:'
    )

    params_rf = [
        ['Parametro', 'Valor'],
        ['n_estimators', '200'],
        ['max_depth', '15'],
        ['min_samples_split', '5'],
        ['min_samples_leaf', '2'],
        ['class_weight', 'balanced'],
        ['random_state', '42'],
    ]
    agregar_tabla_estilizada(doc, params_rf[0], params_rf[1:],
                              col_widths=[5, 5])

    doc.add_heading('Metricas del modelo base', level=2)

    metricas_base_tabla = [
        ['Metrica', 'Valor'],
        ['Accuracy', f"{base['accuracy']:.4f}"],
        ['Precision (desocupado)', f"{base['precision_desocupado']:.4f}"],
        ['Recall (desocupado)', f"{base['recall_desocupado']:.4f}"],
        ['F1-Score (desocupado)', f"{base['f1_desocupado']:.4f}"],
        ['F1 Macro', f"{base['f1_macro']:.4f}"],
        ['N variables', f"{base['n_variables']}"],
        ['Tiempo entrenamiento', f"{base['tiempo_entrenamiento']:.3f} s"],
    ]
    agregar_tabla_estilizada(doc, metricas_base_tabla[0], metricas_base_tabla[1:],
                              col_widths=[6, 6])

    doc.add_heading('Matriz de confusion - Modelo Base', level=2)

    agregar_parrafo(doc,
        f'La matriz de confusion del modelo base muestra que el modelo identifica '
        f'correctamente {mc_base[1][1]} de los {mc_base[1][0] + mc_base[1][1]} desocupados reales '
        f'(Recall = {base["recall_desocupado"]:.4f}), pero genera {mc_base[0][1]} falsos positivos.',
        size=10
    )

    mc_base_tabla = [
        ['', 'Pred. Ocupado', 'Pred. Desocupado'],
        ['Real Ocupado', f'{mc_base[0][0]:,} (VN)', f'{mc_base[0][1]:,} (FP)'],
        ['Real Desocupado', f'{mc_base[1][0]:,} (FN)', f'{mc_base[1][1]:,} (VP)'],
    ]
    agregar_tabla_estilizada(doc, mc_base_tabla[0], mc_base_tabla[1:],
                              col_widths=[4, 4, 4])

    doc.add_paragraph()
    agregar_grafico(doc, datos['graficos'].get('matriz_base'),
                     titulo='Figura 3: Matriz de confusion - Modelo Base')

    doc.add_heading('Importancia de variables', level=2)
    agregar_parrafo(doc,
        'La importancia de Gini permite identificar cuales variables contribuyen mas '
        'a las decisiones del modelo. Las variables mas importantes del modelo base son:',
        size=10
    )
    agregar_grafico(doc, datos['graficos'].get('importancia_vars'),
                     titulo='Figura 4: Importancia de variables (Gini) - Top 20')

    doc.add_page_break()

    # ============================================================
    # 5. ALGORITMO GENETICO
    # ============================================================
    doc.add_heading('5. Algoritmo Genetico', level=1)

    agregar_parrafo(doc,
        'Se implemento un Algoritmo Genetico (AG) utilizando la libreria DEAP para '
        f'resolver el problema de seleccion de variables. El AG opera sobre una '
        f'poblacion de cromosomas binarios de longitud {ag["n_variables_total"]}, '
        f'donde cada gen representa una variable predictora codificada '
        f'(1 = seleccionada, 0 = no seleccionada).'
    )

    doc.add_heading('Funcion de fitness', level=2)
    agregar_parrafo(doc,
        'La funcion de fitness combina el desempeno predictivo con una '
        'penalizacion por complejidad:',
        size=10
    )
    agregar_parrafo(doc,
        'Fitness = F1_desocupado(CV) - penalizacion * (n_variables / n_total)',
        bold=True, size=11,
        alignment=WD_ALIGN_PARAGRAPH.CENTER,
        color=RGBColor(0x2B, 0x57, 0x9A)
    )
    agregar_parrafo(doc,
        'Donde F1_desocupado(CV) es el F1-score promedio de la clase desocupada obtenido '
        'mediante validacion cruzada estratificada de 5 folds sobre el conjunto de '
        'entrenamiento. La penalizacion fomenta soluciones mas parsimoniosas.',
        size=10
    )

    doc.add_heading('Parametros del AG', level=2)

    params_desc = {
        'tamano_poblacion': 'Tamano de poblacion',
        'numero_generaciones': 'Numero de generaciones',
        'probabilidad_cruce': 'Prob. de cruce (dos puntos)',
        'probabilidad_mutacion': 'Prob. de mutacion (bit flip)',
        'tamano_torneo': 'Tamano de torneo (seleccion)',
        'numero_elite': 'Elitismo (mejores directos)',
        'penalizacion_variables': 'Penalizacion por variable',
    }
    params_tabla = [['Parametro', 'Valor']]
    for clave, valor in params.items():
        nombre = params_desc.get(clave, clave)
        params_tabla.append([nombre, str(valor)])
    params_tabla.append(['Validacion cruzada', '5 folds estratificados'])

    agregar_tabla_estilizada(doc, params_tabla[0], params_tabla[1:],
                              col_widths=[7, 5])

    doc.add_heading('Proceso evolutivo', level=2)

    agregar_parrafo(doc,
        'Los siguientes graficos muestran la evolucion del AG a lo largo de las '
        f'{params.get("numero_generaciones", 30)} generaciones. Se observa como el '
        f'fitness del mejor individuo converge hacia el valor final de '
        f'{ag["mejor_fitness"]:.4f}.',
        size=10
    )

    agregar_grafico(doc, datos['graficos'].get('evolucion_fitness'),
                     titulo='Figura 5: Evolucion del fitness a lo largo de las generaciones')
    doc.add_paragraph()
    agregar_grafico(doc, datos['graficos'].get('evolucion_variables'),
                     titulo='Figura 6: Evolucion de la cantidad de variables seleccionadas')

    doc.add_heading('Resultados del AG', level=2)

    resultados_ag = [
        ['Indicador', 'Valor'],
        ['Mejor fitness alcanzado', f"{ag['mejor_fitness']:.4f}"],
        ['Variables seleccionadas', f"{ag['n_variables_seleccionadas']} de {ag['n_variables_total']}"],
        ['Reduccion de dimensionalidad', f"{reduccion_vars:.1f}%"],
        ['Tiempo de evolucion', f"{ag['tiempo_evolucion']:.1f} s ({tiempo_evol_min:.1f} min)"],
    ]
    agregar_tabla_estilizada(doc, resultados_ag[0], resultados_ag[1:],
                              col_widths=[6, 6])

    doc.add_page_break()

    # ============================================================
    # 6. MODELO OPTIMIZADO
    # ============================================================
    doc.add_heading('6. Modelo Optimizado', level=1)

    agregar_parrafo(doc,
        'Se entreno un segundo modelo Random Forest (con identicos hiperparametros) '
        f'utilizando unicamente las {opt["n_variables"]} variables seleccionadas por '
        f'el Algoritmo Genetico. El modelo se evaluo sobre el mismo conjunto de prueba '
        f'que el modelo base, garantizando una comparacion justa.'
    )

    doc.add_heading('Metricas del modelo optimizado', level=2)

    metricas_opt_tabla = [
        ['Metrica', 'Valor'],
        ['Accuracy', f"{opt['accuracy']:.4f}"],
        ['Precision (desocupado)', f"{opt['precision_desocupado']:.4f}"],
        ['Recall (desocupado)', f"{opt['recall_desocupado']:.4f}"],
        ['F1-Score (desocupado)', f"{opt['f1_desocupado']:.4f}"],
        ['F1 Macro', f"{opt['f1_macro']:.4f}"],
        ['N variables', f"{opt['n_variables']}"],
        ['Tiempo entrenamiento', f"{opt['tiempo_entrenamiento']:.3f} s"],
    ]
    agregar_tabla_estilizada(doc, metricas_opt_tabla[0], metricas_opt_tabla[1:],
                              col_widths=[6, 6])

    doc.add_heading('Matriz de confusion - Modelo Optimizado', level=2)

    agregar_parrafo(doc,
        f'Con las variables seleccionadas por el AG, el modelo optimizado identifica '
        f'correctamente {mc_opt[1][1]} desocupados (Recall = {opt["recall_desocupado"]:.4f}) '
        f'y clasifica correctamente {mc_opt[0][0]:,} ocupados.',
        size=10
    )

    mc_opt_tabla = [
        ['', 'Pred. Ocupado', 'Pred. Desocupado'],
        ['Real Ocupado', f'{mc_opt[0][0]:,} (VN)', f'{mc_opt[0][1]:,} (FP)'],
        ['Real Desocupado', f'{mc_opt[1][0]:,} (FN)', f'{mc_opt[1][1]:,} (VP)'],
    ]
    agregar_tabla_estilizada(doc, mc_opt_tabla[0], mc_opt_tabla[1:],
                              col_widths=[4, 4, 4])

    doc.add_paragraph()
    agregar_grafico(doc, datos['graficos'].get('matriz_opt'),
                     titulo='Figura 7: Matriz de confusion - Modelo Optimizado')

    doc.add_page_break()

    # ============================================================
    # 7. COMPARACION DE MODELOS
    # ============================================================
    doc.add_heading('7. Comparacion de Modelos', level=1)

    agregar_parrafo(doc,
        'La siguiente tabla presenta la comparacion detallada entre ambos modelos. '
        'Ambos fueron evaluados sobre exactamente el mismo conjunto de prueba '
        '(20% de los datos, con estratificacion).'
    )

    if 'comparacion' in datos:
        enc = ['Metrica', 'Modelo Base', 'Modelo Optimizado', 'Diferencia']
        filas = []
        for fila in datos['comparacion']:
            metrica = fila.get('Metrica', fila.get('\ufeffMetrica', ''))
            val_base = fila.get('Modelo Base', '')
            val_opt = fila.get('Modelo Optimizado', '')
            diff = fila.get('Diferencia', '')
            try:
                diff_val = float(diff)
                if 'variables' in metrica.lower():
                    filas.append([metrica, f"{int(float(val_base))}",
                                  f"{int(float(val_opt))}", f"{int(diff_val)}"])
                elif 'tiempo' in metrica.lower():
                    filas.append([metrica, f"{float(val_base):.3f}",
                                  f"{float(val_opt):.3f}", f"{diff_val:+.3f}"])
                else:
                    filas.append([metrica, f"{float(val_base):.4f}",
                                  f"{float(val_opt):.4f}", f"{diff_val:+.4f}"])
            except (ValueError, TypeError):
                filas.append([metrica, val_base, val_opt, diff])

        agregar_tabla_estilizada(doc, enc, filas, col_widths=[5, 3, 3.5, 3])

    doc.add_paragraph()
    agregar_grafico(doc, datos['graficos'].get('comparacion_metricas'),
                     titulo='Figura 8: Comparacion visual de metricas entre modelos')

    doc.add_heading('Analisis de la comparacion', level=2)

    # F1
    if diff_f1 > 0.01:
        agregar_bullet(doc, f' el AG MEJORO el F1 de desocupados en {diff_f1:+.4f}.',
                        bold_prefix='F1-Score (desocupado):')
    elif diff_f1 > -0.01:
        agregar_bullet(doc,
            f' el AG MANTUVO el F1 de desocupados (diferencia: {diff_f1:+.6f}, despreciable).',
            bold_prefix='F1-Score (desocupado):')
    else:
        agregar_bullet(doc, f' el AG REDUJO el F1 de desocupados en {diff_f1:+.4f}.',
                        bold_prefix='F1-Score (desocupado):')

    # Recall
    if diff_recall > 0.01:
        agregar_bullet(doc, f' mejoro en {diff_recall:+.4f}.',
                        bold_prefix='Recall (desocupado):')
    elif diff_recall > -0.01:
        agregar_bullet(doc, f' se mantuvo (diferencia: {diff_recall:+.4f}).',
                        bold_prefix='Recall (desocupado):')
    else:
        agregar_bullet(doc,
            f' se redujo en {diff_recall:+.4f}. Esto indica que el modelo optimizado '
            f'deja de detectar algunos desocupados reales.',
            bold_prefix='Recall (desocupado):')

    # Accuracy
    agregar_bullet(doc,
        f' el accuracy paso de {base["accuracy"]:.4f} a {opt["accuracy"]:.4f} '
        f'({diff_accuracy:+.4f}), indicando una leve {"mejora" if diff_accuracy > 0 else "reduccion"} '
        f'en la clasificacion general.',
        bold_prefix='Accuracy:')

    # Variables
    agregar_bullet(doc,
        f' se redujo de {base["n_variables"]} a {opt["n_variables"]} variables '
        f'({reduccion_vars:.1f}% de reduccion), logrando un modelo significativamente '
        f'mas parsimonioso e interpretable.',
        bold_prefix='Reduccion de dimensionalidad:')

    # F1 Macro
    agregar_bullet(doc,
        f' paso de {base["f1_macro"]:.4f} a {opt["f1_macro"]:.4f} ({diff_f1_macro:+.4f}), '
        f'indicando un balance {"mejorado" if diff_f1_macro > 0 else "similar"} entre clases.',
        bold_prefix='F1 Macro:')

    doc.add_page_break()

    # ============================================================
    # 8. SELECCION DE VARIABLES
    # ============================================================
    doc.add_heading('8. Seleccion de Variables', level=1)

    agregar_parrafo(doc,
        f'El Algoritmo Genetico selecciono {ag["n_variables_seleccionadas"]} de las '
        f'{ag["n_variables_total"]} variables codificadas disponibles. A continuacion '
        f'se detalla la seleccion agrupada por variable original del diseno de registro EPH.'
    )

    if 'variables' in datos:
        # Agrupar por variable original
        from collections import OrderedDict
        agrupadas = OrderedDict()
        for v in datos['variables']:
            orig = v.get('Variable original', '')
            if orig not in agrupadas:
                agrupadas[orig] = {'total': 0, 'seleccionadas': 0,
                                    'desc': v.get('Descripcion', ''),
                                    'tipo': v.get('Tipo', '')}
            agrupadas[orig]['total'] += 1
            if v.get('Seleccionada', 'No') == 'Si':
                agrupadas[orig]['seleccionadas'] += 1

        enc_vars = ['Variable', 'Descripcion', 'Tipo', 'Seleccionadas / Total', '% Seleccion']
        filas_vars = []
        for var, info in agrupadas.items():
            pct = (info['seleccionadas'] / info['total'] * 100) if info['total'] > 0 else 0
            filas_vars.append([
                var, info['desc'], info['tipo'],
                f"{info['seleccionadas']} / {info['total']}",
                f"{pct:.0f}%"
            ])

        agregar_tabla_estilizada(doc, enc_vars, filas_vars,
                                  col_widths=[2.5, 4, 2, 3, 2])

    doc.add_heading('Variables seleccionadas (detalle)', level=2)

    agregar_parrafo(doc,
        'Lista completa de las variables codificadas seleccionadas por el AG:',
        size=10
    )

    # Listar variables seleccionadas
    vars_sel = ag.get('variables_seleccionadas', [])
    # Mostrar en columnas de 3
    for i in range(0, len(vars_sel), 3):
        grupo = vars_sel[i:i+3]
        texto = '    '.join([f"- {v}" for v in grupo])
        agregar_parrafo(doc, texto, size=9,
                         alignment=WD_ALIGN_PARAGRAPH.LEFT,
                         space_after=Pt(1))

    doc.add_page_break()

    # ============================================================
    # 9. CONCLUSIONES
    # ============================================================
    doc.add_heading('9. Conclusiones', level=1)

    if reduccion_vars > 0 and diff_f1 >= -0.01:
        agregar_parrafo(doc,
            'El Algoritmo Genetico demostro ser una herramienta efectiva para la '
            'seleccion de variables en este problema de clasificacion. Los principales '
            'hallazgos son:',
        )
    else:
        agregar_parrafo(doc,
            'El Algoritmo Genetico fue aplicado como herramienta de seleccion de '
            'variables. Los principales hallazgos son:',
        )

    agregar_bullet(doc,
        f' el AG logro reducir la dimensionalidad en un {reduccion_vars:.1f}%, '
        f'pasando de {ag["n_variables_total"]} a {ag["n_variables_seleccionadas"]} '
        f'variables predictoras.',
        bold_prefix='Reduccion de dimensionalidad:')

    agregar_bullet(doc,
        f' el F1-score de la clase desocupada se mantuvo practicamente identico '
        f'({base["f1_desocupado"]:.4f} vs. {opt["f1_desocupado"]:.4f}), lo cual '
        f'indica que las variables eliminadas aportaban informacion redundante.',
        bold_prefix='Desempeno predictivo:')

    agregar_bullet(doc,
        f' el accuracy general mejoro levemente de {base["accuracy"]:.4f} a '
        f'{opt["accuracy"]:.4f}, sugiriendo que la eliminacion de variables ruidosas '
        f'puede beneficiar la clasificacion.',
        bold_prefix='Accuracy:')

    agregar_bullet(doc,
        f' el modelo optimizado utiliza {reduccion_vars:.0f}% menos variables, '
        f'lo cual facilita la interpretacion y reduce el riesgo de sobreajuste.',
        bold_prefix='Interpretabilidad:')

    agregar_bullet(doc,
        f' el desbalance de clases ({tasa_desoc:.1f}% de desocupados) sigue '
        f'siendo un factor limitante. Se podrian explorar tecnicas adicionales como '
        f'SMOTE, submuestreo o modelos especializados en datos desbalanceados.',
        bold_prefix='Limitaciones:')

    doc.add_heading('Recomendaciones para trabajo futuro', level=2)

    recomendaciones = [
        'Explorar otros algoritmos de clasificacion (XGBoost, LightGBM, redes neuronales) '
        'para comparar con Random Forest.',
        'Aplicar tecnicas de sobremuestreo (SMOTE) o submuestreo para abordar el '
        'desbalance de clases de manera mas agresiva.',
        'Ajustar los hiperparametros del AG (mas generaciones, mayor poblacion, '
        'diferentes funciones de fitness) para mejorar la seleccion.',
        'Incorporar variables adicionales de la EPH (ingresos, rama de actividad, '
        'intensidad horaria) para enriquecer el modelo.',
        'Validar los resultados con datos de otros trimestres para evaluar la '
        'estabilidad temporal del modelo.',
    ]
    for rec in recomendaciones:
        agregar_bullet(doc, rec, size=10)

    # ============================================================
    # PIE DE PAGINA
    # ============================================================
    doc.add_paragraph()
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('---')
    run.font.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)

    agregar_parrafo(doc,
        f'Reporte generado automaticamente el {datetime.now().strftime("%d/%m/%Y a las %H:%M")}.\n'
        f'Fuente de datos: INDEC - Encuesta Permanente de Hogares (EPH) - '
        f'Total Urbano - 3er Trimestre 2025.',
        size=8, italic=True,
        color=RGBColor(0x99, 0x99, 0x99),
        alignment=WD_ALIGN_PARAGRAPH.CENTER
    )

    return doc


def main():
    print("=" * 60)
    print("  GENERADOR DE REPORTE DOCX")
    print("  Documento Word con analisis y graficas")
    print("=" * 60)

    print("\n[*] Cargando datos de resultados...")
    datos = cargar_todos_los_datos()

    print("\n[*] Generando documento DOCX...")
    doc = generar_documento(datos)

    ruta_salida = os.path.join(RUTA_SALIDA, 'reporte_resultados.docx')
    doc.save(ruta_salida)

    tamano = os.path.getsize(ruta_salida) / 1024
    print(f"\n[OK] Reporte DOCX generado exitosamente:")
    print(f"   Archivo: {ruta_salida}")
    print(f"   Tamano: {tamano:.1f} KB")
    print("=" * 60)


if __name__ == '__main__':
    main()
