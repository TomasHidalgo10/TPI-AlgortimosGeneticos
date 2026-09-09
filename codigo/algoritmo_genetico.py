"""
Módulo del Algoritmo Genético para selección de variables.

Este es el componente central del Trabajo Práctico. Implementa un
Algoritmo Genético utilizando la librería DEAP para buscar el
subconjunto óptimo de variables predictoras que maximice el
F1-score de la clase desocupada.

Representación cromosómica:
    Cada individuo es un vector binario de longitud igual al número
    de variables codificadas. Un 1 indica que la variable se utiliza,
    un 0 indica que no se utiliza.

    Ejemplo: [1, 0, 1, 1, 0, 0, 1, ...]

Función de fitness:
    fitness = F1_desocupados - penalización_por_cantidad_de_variables

    La penalización es pequeña para favorecer modelos más simples
    sin sacrificar significativamente el desempeño.
"""

import random
import numpy as np
import time
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer

from codigo.configuracion import (
    PARAMETROS_AG, PARAMETROS_RF, SEMILLA, FOLDS_CV
)


def crear_funcion_fitness(X_entrenamiento, y_entrenamiento, parametros_rf, folds_cv, penalizacion):
    """
    Crea la función de fitness para el Algoritmo Genético.

    La función evalúa un individuo (subconjunto de variables) mediante:
    1. Seleccionar las variables indicadas por el cromosoma.
    2. Entrenar un Random Forest con validación cruzada estratificada.
    3. Calcular el F1-score promedio de la clase desocupada.
    4. Aplicar una penalización por la cantidad de variables.

    IMPORTANTE: Se utiliza validación cruzada estratificada de K folds
    EXCLUSIVAMENTE sobre el conjunto de entrenamiento. El conjunto de
    prueba NO participa en ningún momento del proceso evolutivo.

    Parámetros
    ----------
    X_entrenamiento : array-like
        Datos de entrenamiento codificados.
    y_entrenamiento : array-like
        Variable objetivo de entrenamiento.
    parametros_rf : dict
        Parámetros del Random Forest.
    folds_cv : int
        Número de folds para validación cruzada.
    penalizacion : float
        Coeficiente de penalización por cantidad de variables.

    Retorna
    -------
    function
        Función de fitness que recibe un individuo y retorna una tupla.
    """
    # Scorer personalizado para F1 de la clase desocupada
    scorer_f1_desocupado = make_scorer(f1_score, pos_label=1, zero_division=0)

    # Validación cruzada estratificada
    cv_estratificada = StratifiedKFold(n_splits=folds_cv, shuffle=True, random_state=SEMILLA)

    def evaluar_individuo(individuo):
        """
        Evalúa un individuo del Algoritmo Genético.

        Parámetros
        ----------
        individuo : list
            Cromosoma binario indicando qué variables usar.

        Retorna
        -------
        tuple
            Tupla con el valor de fitness (requerido por DEAP).
        """
        # Obtener índices de variables seleccionadas
        indices = [i for i, gen in enumerate(individuo) if gen == 1]

        # Si no se selecciona ninguna variable, fitness muy bajo
        if len(indices) == 0:
            return (0.0,)

        # Seleccionar columnas del individuo
        X_seleccionado = X_entrenamiento[:, indices]

        # Crear modelo para evaluación
        modelo_cv = RandomForestClassifier(
            n_estimators=parametros_rf.get('n_estimators', 100),
            max_depth=parametros_rf.get('max_depth', 15),
            min_samples_split=parametros_rf.get('min_samples_split', 5),
            min_samples_leaf=parametros_rf.get('min_samples_leaf', 2),
            class_weight='balanced',
            random_state=SEMILLA,
            n_jobs=-1
        )

        try:
            # Validación cruzada estratificada sobre datos de entrenamiento
            scores = cross_val_score(
                modelo_cv, X_seleccionado, y_entrenamiento,
                cv=cv_estratificada, scoring=scorer_f1_desocupado
            )

            f1_promedio = scores.mean()

            # Penalización por cantidad de variables
            # Se aplica una penalización pequeña para favorecer modelos más simples
            n_variables = len(indices)
            n_total = len(individuo)
            penalizacion_total = penalizacion * (n_variables / n_total)

            fitness = f1_promedio - penalizacion_total

            return (max(fitness, 0.0),)

        except Exception:
            return (0.0,)

    return evaluar_individuo


def configurar_algoritmo_genetico(n_variables, parametros_ag):
    """
    Configura el Algoritmo Genético utilizando DEAP.

    Componentes del AG:
    - Representación: Cromosoma binario
    - Fitness: Maximización (F1 de desocupados)
    - Selección: Torneo
    - Cruce: Cruce de dos puntos
    - Mutación: Bit flip
    - Elitismo: Se preservan los mejores individuos

    Parámetros
    ----------
    n_variables : int
        Número total de variables (longitud del cromosoma).
    parametros_ag : dict
        Parámetros del Algoritmo Genético.

    Retorna
    -------
    deap.base.Toolbox
        Toolbox configurado con todos los operadores.
    """
    # Limpiar registros previos de DEAP si existen
    if 'FitnessMax' in creator.__dict__:
        del creator.FitnessMax
    if 'Individuo' in creator.__dict__:
        del creator.Individuo

    # Definir tipo de fitness (maximización)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    # Definir individuo como lista con fitness asociado
    creator.create("Individuo", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Atributo: gen binario (0 o 1)
    toolbox.register("attr_bool", random.randint, 0, 1)

    # Individuo: cromosoma binario de longitud n_variables
    toolbox.register(
        "individuo",
        tools.initRepeat,
        creator.Individuo,
        toolbox.attr_bool,
        n=n_variables
    )

    # Población: lista de individuos
    toolbox.register("poblacion", tools.initRepeat, list, toolbox.individuo)

    # Operadores genéticos
    toolbox.register("seleccion", tools.selTournament,
                     tournsize=parametros_ag.get('tamano_torneo', 3))
    toolbox.register("cruce", tools.cxTwoPoint)
    toolbox.register("mutacion", tools.mutFlipBit,
                     indpb=1.0/n_variables)  # Probabilidad por bit

    print(f"\n=== CONFIGURACIÓN DEL ALGORITMO GENÉTICO ===")
    print(f"Longitud del cromosoma: {n_variables} genes")
    print(f"Cada gen representa una variable predictora codificada")
    print(f"  1 = variable seleccionada")
    print(f"  0 = variable no seleccionada")
    print(f"\nOperadores:")
    print(f"  Selección: Torneo (tamaño {parametros_ag.get('tamano_torneo', 3)})")
    print(f"  Cruce: Dos puntos (prob. {parametros_ag.get('probabilidad_cruce', 0.8)})")
    print(f"  Mutación: Bit flip (prob. {parametros_ag.get('probabilidad_mutacion', 0.1)})")
    print(f"  Elitismo: {parametros_ag.get('numero_elite', 2)} mejores individuos")
    print(f"\nPoblación: {parametros_ag.get('tamano_poblacion', 30)} individuos")
    print(f"Generaciones: {parametros_ag.get('numero_generaciones', 30)}")

    return toolbox


def ejecutar_algoritmo_genetico(X_entrenamiento, y_entrenamiento, nombres_columnas):
    """
    Ejecuta el Algoritmo Genético completo para selección de variables.

    El AG busca el subconjunto de variables que maximice el F1-score
    de la clase desocupada mediante validación cruzada estratificada
    sobre los datos de entrenamiento.

    Parámetros
    ----------
    X_entrenamiento : array-like
        Datos de entrenamiento codificados.
    y_entrenamiento : array-like
        Variable objetivo de entrenamiento.
    nombres_columnas : list
        Nombres de las columnas codificadas.

    Retorna
    -------
    dict
        Diccionario con resultados del AG incluyendo mejor individuo,
        variables seleccionadas y registro de la evolución.
    """
    n_variables = X_entrenamiento.shape[1]
    parametros_ag = PARAMETROS_AG

    print("\n" + "=" * 60)
    print("  ALGORITMO GENÉTICO - SELECCIÓN DE VARIABLES")
    print("=" * 60)

    # Configurar semilla para reproducibilidad
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)

    # Configurar AG
    toolbox = configurar_algoritmo_genetico(n_variables, parametros_ag)

    # Crear función de fitness
    funcion_fitness = crear_funcion_fitness(
        X_entrenamiento, y_entrenamiento,
        PARAMETROS_RF, FOLDS_CV,
        parametros_ag['penalizacion_variables']
    )
    toolbox.register("evaluar", funcion_fitness)

    # Crear población inicial
    poblacion = toolbox.poblacion(n=parametros_ag['tamano_poblacion'])

    # Estadísticas para seguimiento
    estadisticas = tools.Statistics(lambda ind: ind.fitness.values)
    estadisticas.register("promedio", np.mean)
    estadisticas.register("maximo", np.max)
    estadisticas.register("minimo", np.min)
    estadisticas.register("desviacion", np.std)

    # Hall of Fame (mejores individuos de todas las generaciones)
    salon_fama = tools.HallOfFame(1)

    # Registro de la evolución
    registro_evolucion = {
        'generacion': [],
        'mejor_fitness': [],
        'fitness_promedio': [],
        'n_variables_mejor': [],
        'mejor_individuo': []
    }

    print(f"\nIniciando evolución...")
    inicio = time.time()

    # Evaluar población inicial
    fitnesses = list(map(toolbox.evaluar, poblacion))
    for ind, fit in zip(poblacion, fitnesses):
        ind.fitness.values = fit

    salon_fama.update(poblacion)

    # Registrar generación 0
    mejor_gen0 = tools.selBest(poblacion, 1)[0]
    n_vars_gen0 = sum(mejor_gen0)
    registro_evolucion['generacion'].append(0)
    registro_evolucion['mejor_fitness'].append(mejor_gen0.fitness.values[0])
    registro_evolucion['fitness_promedio'].append(
        np.mean([ind.fitness.values[0] for ind in poblacion])
    )
    registro_evolucion['n_variables_mejor'].append(n_vars_gen0)
    registro_evolucion['mejor_individuo'].append(list(mejor_gen0))

    print(f"  Gen  0 | Mejor: {mejor_gen0.fitness.values[0]:.4f} | "
          f"Prom: {registro_evolucion['fitness_promedio'][-1]:.4f} | "
          f"Vars: {n_vars_gen0}")

    # Evolución por generaciones
    for gen in range(1, parametros_ag['numero_generaciones'] + 1):
        # Seleccionar la siguiente generación
        descendencia = toolbox.seleccion(poblacion, len(poblacion) - parametros_ag['numero_elite'])
        descendencia = list(map(toolbox.clone, descendencia))

        # Aplicar cruce
        for hijo1, hijo2 in zip(descendencia[::2], descendencia[1::2]):
            if random.random() < parametros_ag['probabilidad_cruce']:
                toolbox.cruce(hijo1, hijo2)
                del hijo1.fitness.values
                del hijo2.fitness.values

        # Aplicar mutación
        for mutante in descendencia:
            if random.random() < parametros_ag['probabilidad_mutacion']:
                toolbox.mutacion(mutante)
                del mutante.fitness.values

        # Evaluar individuos sin fitness (nuevos o modificados)
        invalidos = [ind for ind in descendencia if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluar, invalidos))
        for ind, fit in zip(invalidos, fitnesses):
            ind.fitness.values = fit

        # Elitismo: añadir los mejores de la generación anterior
        elite = tools.selBest(poblacion, parametros_ag['numero_elite'])
        descendencia.extend(elite)

        # Reemplazar población
        poblacion[:] = descendencia

        # Actualizar Hall of Fame
        salon_fama.update(poblacion)

        # Registrar estadísticas de la generación
        mejor = tools.selBest(poblacion, 1)[0]
        n_vars = sum(mejor)
        fitness_prom = np.mean([ind.fitness.values[0] for ind in poblacion])

        registro_evolucion['generacion'].append(gen)
        registro_evolucion['mejor_fitness'].append(mejor.fitness.values[0])
        registro_evolucion['fitness_promedio'].append(fitness_prom)
        registro_evolucion['n_variables_mejor'].append(n_vars)
        registro_evolucion['mejor_individuo'].append(list(mejor))

        # Mostrar progreso cada 5 generaciones
        if gen % 5 == 0 or gen == parametros_ag['numero_generaciones']:
            print(f"  Gen {gen:>2} | Mejor: {mejor.fitness.values[0]:.4f} | "
                  f"Prom: {fitness_prom:.4f} | Vars: {n_vars}")

    tiempo_evolucion = time.time() - inicio

    # Mejor individuo encontrado
    mejor_individuo = salon_fama[0]
    indices_seleccionados = [i for i, gen in enumerate(mejor_individuo) if gen == 1]
    variables_seleccionadas = [nombres_columnas[i] for i in indices_seleccionados]

    print(f"\n{'='*60}")
    print(f"  RESULTADO DEL ALGORITMO GENÉTICO")
    print(f"{'='*60}")
    print(f"  Tiempo total: {tiempo_evolucion:.1f} segundos")
    print(f"  Mejor fitness: {mejor_individuo.fitness.values[0]:.4f}")
    print(f"  Variables totales: {n_variables}")
    print(f"  Variables seleccionadas: {len(variables_seleccionadas)}")
    print(f"  Reducción: {(1 - len(variables_seleccionadas)/n_variables)*100:.1f}%")
    print(f"\n  Variables seleccionadas:")
    for var in variables_seleccionadas:
        print(f"    ✓ {var}")

    return {
        'mejor_individuo': list(mejor_individuo),
        'mejor_fitness': mejor_individuo.fitness.values[0],
        'indices_seleccionados': indices_seleccionados,
        'variables_seleccionadas': variables_seleccionadas,
        'registro_evolucion': registro_evolucion,
        'tiempo_evolucion': tiempo_evolucion,
        'n_variables_total': n_variables,
        'n_variables_seleccionadas': len(variables_seleccionadas),
        'parametros_ag': parametros_ag
    }
