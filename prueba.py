# from scipy.stats import mannwhitneyu

# x = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
# y = [9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8]

# valor = mannwhitneyu(x,y,alternative='less')

# print(valor)

from Problem.SCP.problem import SCP

instance = SCP('scp41')

print(instance.getColumns())