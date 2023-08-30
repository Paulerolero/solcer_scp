import warnings

warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np
import random


import numpy as np

def iterarAnaconda(iter, poblacion, columns, fitness, bestSolutions, BestFitnessArray):
    N = len(poblacion)
    m = columns
    lb = 0
    ub = 1

    r = np.random.uniform(0, 1, (N, m))
    I = np.random.choice([1, 2], (N, m))

    CFL_i = bestSolutions
    CFF_i = BestFitnessArray
    CFF_i_max =  max(CFF_i) if CFF_i else 0


    for i in range(N):
        PC_i=[]
        for j in CFF_i:
            PC_i_j = (j - CFF_i_max) / (sum([CFF_n - CFF_i_max for CFF_n in CFF_i]))
            PC_i.append(PC_i_j)
        
            # C_i = PC_i_j-C_i[j-1] 

    #     selected_index = next((j for j, C_j in enumerate(C_i) if r[i, j] < C_j), -1)
    #     SF_i = CFL_i[selected_index] if selected_index != -1 else np.zeros(m)

    #     x_P1 = poblacion[i] + r[i] * (SF_i - I[i] * poblacion[i])

    #     if sum(x_P1) < sum(Best):
    #         poblacion[i] = x_P1

    #     x_P2 = poblacion[i] + (1 - 2 * r[i]) * (ub - lb) / iter


    #     if sum(x_P2) < sum(Best):
    #         poblacion[i] = x_P2

    # return poblacion