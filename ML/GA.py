#-*- coding: UTF-8 -*-

import random
import math
import numpy as np
import matplotlib.pyplot as plt
 

population_size = 500  # 种群数量
generations = 200  # 迭代次数
chrom_length = 10   # 染色体长度
pc = 0.6  # 交配概率
pm = 0.01  # 变异概率
genetic_population = []  # 种群的基因编码（二进制）
population = []  # 种群对应的十进制数值,并标准化范围到[0, 10]
fitness = []  # 适应度
fitness_mean = []
optimum_solution = []  # 每次迭代的所获得的最优解
 
 
# 为染色体进行0,1编码，生成初始种群
def chrom_encoding():
    for i in range(population_size):
        population_i = []
        for j in range(chrom_length):
            population_i.append(random.randint(0, 1))
        genetic_population.append(population_i)
 
 
# 对染色体进行解码，将二进制转化为十进制
def chrom_decoding():
    population.clear()
    for i in range(population_size):
        value = 0
        for j in range(chrom_length):
            value += genetic_population[i][j] * (2 ** (chrom_length - 1 - j))
        population.append(value * 10 / (2 ** (chrom_length) - 1))
 
 
# 计算每个染色体的适应度
def calculate_fitness():
    sum = 0.0
    fitness.clear()
    for i in range(population_size):
        function_value = 10 * math.sin(5 * population[i]) + 7 * math.cos(4 * population[i])
        if function_value > 0.0:
            sum += function_value
            fitness.append(function_value)
        else:
            fitness.append(0.0)
    # 返回群体的平均适应度
    return sum / population_size
 
 
# 获取最大适应度的个体和对应的编号
def best_value():
    max_fitness = fitness[0]
    max_chrom = 0
    for i in range(population_size):
        if fitness[i] > max_fitness:
            max_fitness = fitness[i]
            max_chrom = i
    return max_chrom, max_fitness
 
 
# 采用轮盘赌算法进行选择过程，重新选择与种群数量相等的新种群
def selection():
    fitness_proportion = []
    fitness_sum = 0
    for i in range(population_size):
        fitness_sum += fitness[i]
    # 计算生存率
    for i in range(population_size):
        fitness_proportion.append(fitness[i] / fitness_sum)
    pie_fitness = []
    cumsum = 0.0
    for i in range(population_size):
        pie_fitness.append(cumsum + fitness_proportion[i])
        cumsum += fitness_proportion[i]
    pie_fitness[-1] = 1
    # 生成随机数在轮盘上选点[0, 1)
    random_selection = []
    for i in range(population_size):
        random_selection.append(random.random())
    random_selection.sort()
    # 选择新种群
    new_genetic_population = []
    random_selection_id = 0
    global genetic_population
    for i in range(population_size):
        while random_selection_id < population_size and random_selection[random_selection_id] < pie_fitness[i]:
            new_genetic_population.append(genetic_population[i])
            random_selection_id += 1
    genetic_population = new_genetic_population
 
'''
# 用numpy的random.choice函数直接模拟轮盘赌算法
def selection():
    fitness_array = np.array(fitness)
    new_population_id = np.random.choice(np.arange(population_size), (population_size,),
                                         replace=True, p=fitness_array/fitness_array.sum())
    new_genetic_population = []
    global genetic_population
    for i in range(population_size):
        new_genetic_population.append(genetic_population[new_population_id[i]])
    genetic_population = new_genetic_population
'''
 
 
# 进行交配过程
def crossover():
    for i in range(0, population_size - 1, 2):
        if random.random() < pc:
            # 随机选择交叉点
            change_point = random.randint(0, chrom_length - 1)
            temp1 = []
            temp2 = []
            temp1.extend(genetic_population[i][0: change_point])
            temp1.extend(genetic_population[i+1][change_point:])
            temp2.extend(genetic_population[i+1][0: change_point])
            temp2.extend(genetic_population[i][change_point:])
            genetic_population[i] = temp1
            genetic_population[i+1] = temp2
 
 
# 进行基因的变异
def mutation():
    for i in range(population_size):
        if random.random() < pm:
            mutation_point = random.randint(0, chrom_length - 1)
            if genetic_population[i][mutation_point] == 0:
                genetic_population[i][mutation_point] = 1
            else:
                genetic_population[i][mutation_point] = 0
 
 
chrom_encoding()
for step in range(generations):
    chrom_decoding()
    fit_mean = calculate_fitness()
    best_id, best_fitness = best_value()
    optimum_solution.append(best_fitness)
    fitness_mean.append(fit_mean)
    selection()
    crossover()
    mutation()
 
# 最优解随迭代次数的变化
fig1 = plt.figure(1)
plt.plot(range(1, generations + 1), optimum_solution)
plt.xlabel('iterations', fontproperties='SimHei')
plt.ylabel('optimal solution', fontproperties='SimHei')
# 平均适应度随迭代次数的变化
fig2 = plt.figure(2)
plt.plot(range(1, generations + 1), fitness_mean)
plt.xlabel('iterations', fontproperties='SimHei')
plt.ylabel('mean fitness', fontproperties='SimHei')
plt.show()
print(optimum_solution)
print(fitness_mean)
print('结束')