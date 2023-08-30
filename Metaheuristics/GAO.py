import warnings

warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np
import random

# def iterarGAO(maxIter, t, dimension, poblacion, bestSolution, l, u):
#   """
#   Implementación del algoritmo Green Anaconda Optimization (GAO)

#   Args:
#     maxIter: El número máximo de iteraciones del algoritmo.
#     t: El número actual de iteración.
#     dimension: El número de dimensiones del problema.
#     poblacion: La población a perturbar.
#     bestSolution: La mejor solución encontrada hasta ahora.

#   Returns:
#     La mejor solución encontrada por el algoritmo.
#     La población perturbada.
#   """

#   # Identificamos las hembras candidatas utilizando la ecuación (4).
#   candidate_females_list = []
#   for i in range(poblacion.shape[0]):
#         for j in range(poblacion.shape[1]):
#             if poblacion[i, j] < bestSolution and j != i:
#                 candidate_females_list.append((i, j))


#   # Calculamos la función de concentración de las hembras candidatas utilizando la ecuación (5).
#   concentration_function = np.zeros(len(candidate_females_list))
#   for i, j in candidate_females_list:
#     concentration_function[i] = candidate_females_list[i][1] - max(candidate_females_list, key=lambda x: x[1])[1]
#     concentration_function[i] /= sum(concentration_function)

#   # Calculamos la función de probabilidad acumulativa de las hembras candidatas utilizando la ecuación (6).
#   cumulative_probability_function = np.zeros(len(candidate_females_list) + 1)
#   cumulative_probability_function[0] = 0
#   for i in range(len(candidate_females_list)):
#     cumulative_probability_function[i + 1] = cumulative_probability_function[i] + concentration_function[i]

#   # Determinamos la hembra seleccionada utilizando la ecuación (7).
#   selected_female = []
#   r = np.random.rand()
#   for i in range(len(cumulative_probability_function)):
#     if cumulative_probability_function[i] <= r < cumulative_probability_function[i + 1]:
#       selected_female = candidate_females_list[i]
#       break

#   # Calculamos la nueva posición del miembro GAO i utilizando la ecuación (8).
#   # new_position_i = poblacion[i] + r * (poblacion[selected_female[0]][selected_female[1]] - poblacion[i][t])
#   # new_position_i = poblacion[i] + r * (poblacion[selected_female[0]][selected_female[1]] - poblacion[i].item(t))
#   new_position_i = poblacion[i][t] + r * (poblacion[selected_female[0]][selected_female[1]] - poblacion[i][t])




#   # Actualizamos el miembro GAO i utilizando la ecuación (9).
#   if bestSolution[i] < poblacion[i][t]:
#     poblacion[i] = new_position_i

#   # Fase 2: estrategia de caza (explotación)

#   # Calculamos la nueva posición del mejor individuo utilizando la ecuación (10).
#   # new_position_best_individual = bestSolution + (1 - 2 * r) * (u - l) * np.sqrt(t)
#   new_position_best_individual = bestSolution + (1 - 2 * r) * (u[0] - l[0]) * np.sqrt(t)

#   # Actualizamos el mejor individuo de la población si la nueva posición es mejor.
#   # if bestSolution[i] < new_position_best_individual:
#   if any(bestSolution[i] < new_position_best_individual):
#     bestSolution = new_position_best_individual

#   # Devolvemos la mejor solución encontrada.
#   return poblacion, bestSolution

# def iterarGAO(maxIter, t, dimension, poblacion, bestSolution, l, u):
#     pop_size = poblacion[0].shape
#     for i in range(pop_size):
#         selected_female = select_female(poblacion, i, bestSolution)

# def select_female(population, idx, bestSolution):

#     females = [ind for i, ind in enumerate(population) if bestSolution < population[idx] and i != idx]

#     # Si no hay individuos más fuertes, devolver el individuo actual
#     if not females:
#         return population[idx]

#     # Calcular probabilidad basada en valor objetivo
#     max_obj_value = max(females, key=lambda x: x.obj_value)
#     probabilities = [female - bestSolution for female in females]

#     if sum(probabilities) == 0:
#         probabilities = [1/len(females) for _ in females]
#     else:
#         probabilities = [p/sum(probabilities) for p in probabilities]

#     # Seleccionar hembra
#     return random.choices(females, probabilities)[0]

import numpy as np

def iterarAnaconda(maxIter, iter, columns, poblacion, Best,fitness):
    N = len(poblacion)
    m = columns
    lb = 0
    ub = 1

    r = np.random.uniform(0, 1, (N, m))
    I = np.random.choice([1, 2], (N, m))

    for t in range(maxIter):
        for i in range(N):
            
            # F_values = [sum(poblacion[k]) for k in range(N)]
            F_values = fitness
            
            CFL_i = [poblacion[k] for k in range(N) if F_values[k] < F_values[i] and k != i]

            CFF_i = [F_values[k] for k in range(N) if F_values[k] < F_values[i] and k != i]
            CFF_max_i = max(CFF_i) if CFF_i else 0

            PC_i = [(CFF_j - CFF_max_i) / (sum([CFF_n - CFF_max_i for CFF_n in CFF_i])) for j, CFF_j in enumerate(CFF_i)]
            
            C_i = np.cumsum(PC_i)

            selected_index = next((j for j, C_j in enumerate(C_i) if r[i, j] < C_j), -1)
            SF_i = CFL_i[selected_index] if selected_index != -1 else np.zeros(m)

            x_P1 = poblacion[i] + r[i] * (SF_i - I[i] * poblacion[i])

            if sum(x_P1) < sum(Best):
                poblacion[i] = x_P1

            x_P2 = poblacion[i] + (1 - 2 * r[i]) * (ub - lb) / iter


            if sum(x_P2) < sum(Best):
                poblacion[i] = x_P2

    return poblacion
