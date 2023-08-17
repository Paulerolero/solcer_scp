import random
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev

class Individual:
    def __init__(self, position, obj_value):
        self.position = position
        self.obj_value = obj_value

def GAO(obj_func, lb, ub, num_var, pop_size, max_iter):
    """Ejecuta el Algoritmo de Optimización Gacela (GAO).

    Args:
        obj_func: Función objetivo a optimizar.
        lb: Límites inferiores para las variables.
        ub: Límites superiores para las variables.
        num_var: Número de variables.
        pop_size: Tamaño de la población.
        max_iter: Número máximo de iteraciones.

    Returns:
        El mejor individuo obtenido por el algoritmo.
    """
    # Inicializar población
    population = [Individual([random.uniform(lb[j], ub[j]) for j in range(num_var)], float('inf')) for i in range(pop_size)]

    for ind in population:
        ind.obj_value = obj_func(ind.position)

    best_solution = min(population, key=lambda x: x.obj_value)
    best_obj_value = float("inf")
    best_values_per_iteration = []

    for t in range(max_iter):
        for i in range(pop_size):
            # Fase 1 - Temporada de apareamiento (exploración)
            selected_female = select_female(population, i)
            new_position = move_towards_female(population[i], selected_female)
            new_position = enforce_bounds(new_position, lb, ub)
            new_obj_value = obj_func(new_position)

            if new_obj_value < population[i].obj_value:
                population[i].position = new_position
                population[i].obj_value = new_obj_value

            # Fase 2 - Estrategia de caza (explotación)
            nearby_position = generate_nearby_position(population[i], lb, ub, t)
            nearby_position = enforce_bounds(nearby_position, lb, ub)
            nearby_obj_value = obj_func(nearby_position)

            if nearby_obj_value < population[i].obj_value:
                population[i].position = nearby_position
                population[i].obj_value = nearby_obj_value

            # Actualizar mejor solución
            if population[i].obj_value < best_solution.obj_value:
                best_solution = population[i]
                best_obj_value = population[i].obj_value

        best_values_per_iteration.append(best_obj_value)

    # Gráfica de los mejores valores de función objetivo a través de las iteraciones
    plt.plot(best_values_per_iteration)
    plt.xlabel('Iteraciones')
    plt.ylabel('Mejor valor de función objetivo')
    plt.title('Evolución del mejor valor de función objetivo')
    plt.show()

    return best_solution

def enforce_bounds(position, lb, ub):
    """Asegura que las posiciones estén dentro de los límites establecidos.

    Args:
        position: Posición del individuo.
        lb: Límites inferiores.
        ub: Límites superiores.

    Returns:
        Posición ajustada según los límites.
    """
    return [min(max(position[i], lb[i]), ub[i]) for i in range(len(position))]

def select_female(population, idx):
    """Selecciona una hembra para el individuo actual basado en valores objetivos.

    Args:
        population: Población actual.
        idx: Índice del individuo actual.

    Returns:
        Individuo hembra seleccionado.
    """
    females = [ind for i, ind in enumerate(population) if ind.obj_value < population[idx].obj_value and i != idx]

    # Si no hay individuos más fuertes, devolver el individuo actual
    if not females:
        return population[idx]

    # Calcular probabilidad basada en valor objetivo
    max_obj_value = max(females, key=lambda x: x.obj_value).obj_value
    probabilities = [female.obj_value - max_obj_value for female in females]

    if sum(probabilities) == 0:
        probabilities = [1/len(females) for _ in females]
    else:
        probabilities = [p/sum(probabilities) for p in probabilities]

    # Seleccionar hembra
    return random.choices(females, probabilities)[0]

def move_towards_female(individual, female):
    """Mueve el individuo actual hacia la hembra seleccionada.

    Args:
        individual: Individuo actual.
        female: Hembra seleccionada.

    Returns:
        Nueva posición para el individuo.
    """
    return [individual.position[d] + random.random() * (female.position[d] - random.choice([1,2]) * individual.position[d]) for d in range(len(individual.position))]

def generate_nearby_position(individual, lb, ub, t):
    """Genera una nueva posición cercana al individuo actual.

    Args:
        individual: Individuo actual.
        lb: Límites inferiores.
        ub: Límites superiores.
        t: Iteración actual.

    Returns:
        Nueva posición cercana al individuo.
    """
    return [individual.position[d] + (1 - 2 * random.random()) * (ub[d] - lb[d]) / (t+1) for d in range(len(individual.position))]
