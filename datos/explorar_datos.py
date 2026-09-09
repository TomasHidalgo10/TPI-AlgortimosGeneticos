
import pandas as pd

datos = pd.read_csv(r'd:\tpi-Ageneticos\datos\originales\personas_tot.urb_3T_2025.txt', sep=';', low_memory=False)

print('=== DIMENSIONES ===')
print(f'Filas: {len(datos)}, Columnas: {len(datos.columns)}')

print('\n=== ESTADO (condicion actividad) ===')
print(datos['ESTADO'].value_counts().sort_index())

print('\n=== CH03 (relacion parentesco) ===')
print(datos['CH03'].value_counts().sort_index())

print('\n=== CH04 (sexo) ===')
print(datos['CH04'].value_counts().sort_index())

print('\n=== CH06 (edad) ===')
ch06 = datos['CH06']
print(f'Min: {ch06.min()}, Max: {ch06.max()}, Media: {ch06.mean():.1f}')
print(ch06.describe())
print('Valores negativos:')
print(ch06[ch06 < 0].value_counts().sort_index())

print('\n=== CH07 (estado civil) ===')
print(datos['CH07'].value_counts().sort_index())

print('\n=== CH08 (cobertura salud) ===')
print(datos['CH08'].value_counts().sort_index())

print('\n=== NIVEL_ED (nivel educativo) ===')
print(datos['NIVEL_ED'].value_counts().sort_index())

print('\n=== AGLOMERADO ===')
print(f'Valores unicos: {datos["AGLOMERADO"].nunique()}')
print(datos['AGLOMERADO'].value_counts().sort_index())

print('\n=== CH09 (asistencia escolar) ===')
print(datos['CH09'].value_counts().sort_index())

print('\n=== CH10 (condicion lectura/escritura) ===')
print(datos['CH10'].value_counts().sort_index())

print('\n=== CH11 (condicion asistencia) ===')
print(datos['CH11'].value_counts().sort_index())

print('\n=== CH12 (ultimo nivel cursado) ===')
print(datos['CH12'].value_counts().sort_index())

print('\n=== CH13 (finalizo nivel) ===')
print(datos['CH13'].value_counts().sort_index())

print('\n=== CH14 (anios cursados) ===')
print(datos['CH14'].value_counts().sort_index().head(20))

print('\n=== CH15 (lugar nacimiento) ===')
print(datos['CH15'].value_counts().sort_index())

print('\n=== CH16 (tipo vivienda) ===')
print(datos['CH16'].value_counts().sort_index())

print('\n=== PROVINCIA ===')
print(f'Valores unicos: {datos["PROVINCIA"].nunique()}')
print(datos['PROVINCIA'].value_counts().sort_index())

# Filtrar PEA
pea = datos[datos['ESTADO'].isin([1, 2])]
print('\n=== PEA FILTRADA ===')
print(f'Total PEA: {len(pea)}')
print(f'Ocupados (ESTADO=1): {(pea["ESTADO"]==1).sum()}')
print(f'Desocupados (ESTADO=2): {(pea["ESTADO"]==2).sum()}')
print(f'% Ocupados: {(pea["ESTADO"]==1).sum()/len(pea)*100:.2f}%')
print(f'% Desocupados: {(pea["ESTADO"]==2).sum()/len(pea)*100:.2f}%')

print('\n=== CAT_OCUP (categoria ocupacional) ===')
print(datos['CAT_OCUP'].value_counts().sort_index())

print('\n=== CAT_INAC (categoria inactividad) ===')
print(datos['CAT_INAC'].value_counts().sort_index())

print('\n=== INTENSI (intensidad laboral) ===')
print(datos['INTENSI'].value_counts().sort_index())
