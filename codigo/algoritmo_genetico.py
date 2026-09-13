"""
Algoritmo Genético para selección de variables.

Busca el subconjunto óptimo de variables que maximiza el
F1-score de la clase desocupada usando DEAP.

Representación:
    Vector binario. 1 = variable usada, 0 = no usada.

Fitness:
    F1_desocupados - penalización_por_cantidad_de_variables
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

    Parámetros
    ----------
    X_entrenamiento : array-like
        Datos de entrenamiento.
    y_entrenamiento : array-like
        Target de entrenamiento.
    parametros_rf : dict
        Parámetros para Random Forest.
    folds_cv : int
        Folds de CV.
    penalizacion : float
        Penalización por cantidad de variables.

    Retorna
    -------
    function
        Función de fitness.
    """
    # Scorer F1 desocupados
    scorer_f1_desocupado = make_scorer(f1_score, pos_label=1, zero_division=0)

    # Validación cruzada estratificada
    cv_estratificada = StratifiedKFold(n_splits=folds_cv, shuffle=True, random_state=SEMILLA)

    def evaluar_individuo(individuo):
        # Obtener variables seleccionadas
        indices = [i for i, gen in enumerate(individuo) if gen == 1]

        # Penalización por no usar variables
        if len(indices) == 0:
            return (0.0,)

        # Seleccionar columnas del individuo
        X_seleccionado = X_entrenamiento[:, indices]

        # Modelo RF
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

            # Penalización por cantidad de vars
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

    Parámetros
    ----------
    n_variables : int
        Total de variables.
    parametros_ag : dict
        Parámetros AG.

    Retorna
    -------
    deap.base.Toolbox
        Toolbox de DEAP configurado.
    """
    if 'FitnessMax' in creator.__dict__:
        del creator.FitnessMax
    if 'Individuo' in creator.__dict__:
        del creator.Individuo

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individuo", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("attr_bool", random.randint, 0, 1)

    toolbox.register(
        "individuo",
        tools.initRepeat,
        creator.Individuo,
        toolbox.attr_bool,
        n=n_variables
    )

    toolbox.register("poblacion", tools.initRepeat, list, toolbox.individuo)

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

    Parámetros
    ----------
    X_entrenamiento : array-like
    y_entrenamiento : array-like
    nombres_columnas : list

    Retorna
    -------
    dict
        Resultados del AG.
    """
    n_variables = X_entrenamiento.shape[1]
    parametros_ag = PARAMETROS_AG

    print("\n" + "=" * 60)
    print("  ALGORITMO GENÉTICO - SELECCIÓN DE VARIABLES")
    print("=" * 60)

    random.seed(SEMILLA)
    np.random.seed(SEMILLA)

    toolbox = configurar_algoritmo_genetico(n_variables, parametros_ag)

    funcion_fitness = crear_funcion_fitness(
        X_entrenamiento, y_entrenamiento,
        PARAMETROS_RF, FOLDS_CV,
        parametros_ag['penalizacion_variables']
    )
    toolbox.register("evaluar", funcion_fitness)

    poblacion = toolbox.poblacion(n=parametros_ag['tamano_poblacion'])

    estadisticas = tools.Statistics(lambda ind: ind.fitness.values)
    estadisticas.register("promedio", np.mean)
    estadisticas.register("maximo", np.max)
    estadisticas.register("minimo", np.min)
    estadisticas.register("desviacion", np.std)

    salon_fama = tools.HallOfFame(1)

    # Log de evolución
    registro_evolucion = {
        'generacion': [],
        'mejor_fitness': [],
        'fitness_promedio': [],
        'n_variables_mejor': [],
        'mejor_individuo': []
    }

    print(f"\nIniciando evolución...")
    inicio = time.time()

    fitnesses = list(map(toolbox.evaluar, poblacion))
    for ind, fit in zip(poblacion, fitnesses):
        ind.fitness.values = fit

    salon_fama.update(poblacion)

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

    for gen in range(1, parametros_ag['numero_generaciones'] + 1):
        descendencia = toolbox.seleccion(poblacion, len(poblacion) - parametros_ag['numero_elite'])
        descendencia = list(map(toolbox.clone, descendencia))

        # Cruce
        for hijo1, hijo2 in zip(descendencia[::2], descendencia[1::2]):
            if random.random() < parametros_ag['probabilidad_cruce']:
                toolbox.cruce(hijo1, hijo2)
                del hijo1.fitness.values
                del hijo2.fitness.values

        # Mutación
        for mutante in descendencia:
            if random.random() < parametros_ag['probabilidad_mutacion']:
                toolbox.mutacion(mutante)
                del mutante.fitness.values

        invalidos = [ind for ind in descendencia if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluar, invalidos))
        for ind, fit in zip(invalidos, fitnesses):
            ind.fitness.values = fit

        elite = tools.selBest(poblacion, parametros_ag['numero_elite'])
        descendencia.extend(elite)

        poblacion[:] = descendencia

        salon_fama.update(poblacion)

        mejor = tools.selBest(poblacion, 1)[0]
        n_vars = sum(mejor)
        fitness_prom = np.mean([ind.fitness.values[0] for ind in poblacion])

        registro_evolucion['generacion'].append(gen)
        registro_evolucion['mejor_fitness'].append(mejor.fitness.values[0])
        registro_evolucion['fitness_promedio'].append(fitness_prom)
        registro_evolucion['n_variables_mejor'].append(n_vars)
        registro_evolucion['mejor_individuo'].append(list(mejor))

        if gen % 5 == 0 or gen == parametros_ag['numero_generaciones']:
            print(f"  Gen {gen:>2} | Mejor: {mejor.fitness.values[0]:.4f} | "
                  f"Prom: {fitness_prom:.4f} | Vars: {n_vars}")

    tiempo_evolucion = time.time() - inicio

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
