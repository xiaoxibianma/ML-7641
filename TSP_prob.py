import matplotlib.pyplot as plt
from ut import ga_parameter_curve, algorithm_compare, \
    mean_time_compare, score_compare, sa_parameter_curve
import mlrose
import numpy as np
import random
import time
import os

question = "Travel Salesman"

# Create the problem
def get_tsp_coords_list(seed_num, coords_num=8):
    random.seed(seed_num)
    coords = []
    for i in range(coords_num):
        coords.append((random.random() * 10, random.random() * 10))
    return coords


# Optimized GA parameter pop size
def find_best_param_ga_pop_size(seed, problem):

    pop_size = [10, 20, 45, 65, 85, 100]

    # convert int to str
    pop_size_string = []
    for size in pop_size:
        pop_size_string.append(str(size))

    fitness_list = []
    iters = [1, 5, 10, 20, 30, 40,  50, 60, 70, 80, 90, 100]
    score = []

    for size in pop_size:
        for iter in iters:
            best_state, best_fitness = mlrose.genetic_alg(problem,
                                                                  max_attempts=10,
                                                                  max_iters=iter,
                                                                  random_state=seed,
                                                                  pop_size=size)
            score.append(best_fitness)
            print("Travel Salesman Genetic Algorithm pop_size Fitness {}".format(best_fitness))

        fitness_list.append(np.mean(score))
        print("when size = ", size, "  the mean score is " , np.mean(score))

    ga_parameter_curve(pop_size_string, fitness_list, "Pop_size", 0.4, "TSP")


# Optimized GA parameter mutation prob
def find_best_param_ga_mutation_prob(seed, problem):
    mutation_prob = [0.1, 0.3, 0.7, 1.0]
    mutation_prob_String = ["0.1", "0.3", "0.7", "1.0"]
    fitness_list = []
    iters = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    score = []

    for prob in mutation_prob:
        for iter in iters:
            best_state, best_fitness = mlrose.genetic_alg(problem,
                                                          max_attempts=10,
                                                          max_iters=iter,
                                                          random_state=seed,
                                                          mutation_prob=prob)
            score.append(best_fitness)
            print("Travel Salesman Genetic Algorithm pop_size Fitness {}".format(best_fitness))

        fitness_list.append(np.mean(score))
        print("when prob = ", prob, "  the mean score is ", np.mean(score))

    ga_parameter_curve(mutation_prob_String, fitness_list, "mutation_prob", 0.4, "TSP")

# Optimized SA parameter decay
def find_best_param_sa_decay(seed, problem):
    init_temp_String = []
    init_temp = [0.1, 0.2, 0.3, 0.5, 0.66, 0.99]
    for int in init_temp:
        init_temp_String.append(str(int))

    fitness_list = []
    iterations = [1, 250, 500, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]
    score = []

    for init in init_temp:
        for iter in iterations:
            schedule2 = mlrose.GeomDecay(init_temp=500, decay=init)
            best_state, best_fitness = mlrose.simulated_annealing(problem,
                                                                  max_attempts=10,
                                                                  max_iters=iter,
                                                                  random_state=seed,
                                                                  schedule=schedule2)
            score.append(best_fitness)

        fitness_list.append(np.mean(score))
        print("when init = ", init, "  the mean score is ", np.mean(score))

    sa_parameter_curve(init_temp_String, fitness_list, "sa_dacay", 0.4, "TSP")

def find_best_param_mimic_keep_cpt(seed, problem):
    keep_cpt_String = []
    keep_cpt = [0.1, 0.5, 0.9]
    for int in keep_cpt:
        keep_cpt_String.append(str(int))

    fitness_list = []
    iterations = [1, 250, 500, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]
    score = []

    for pct in keep_cpt:
        for iter in iterations:
            best_state, best_fitness = mlrose.mimic(problem,
                                                    max_attempts=10,
                                                    max_iters=iter,
                                                    random_state=seed,
                                                    keep_pct=pct)
            score.append(best_fitness)

        fitness_list.append(np.mean(score))
        print("when pct = ", pct, "  the mean score is ", np.mean(score))

    sa_parameter_curve(keep_cpt_String, fitness_list, "keep_pct", 0.4, "TSP")


# compare RHC, SA, GA, MIMIC
def process_tsp_with_four_algo(coor, seed, problem):

    iterations = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    iterations.extend([i for i in range(100, 3000, 50)])

    fitness_list = []
    mean_fitness_list = []
    mean_time_list = []

    #===================================================================================
    # =============================  R H C==============================================
    # ==================================================================================

    fitness_score_RHC = []
    time_score_RHC = []

    for iter in iterations:
        start_time = time.time()
        best_state, best_fitness = mlrose.random_hill_climb(problem,
                                                            max_attempts=10,
                                                            max_iters=iter,
                                                            random_state=seed)
        end_time = time.time()

        time_total = end_time - start_time
        fitness_score_RHC.append(best_fitness)
        time_score_RHC.append(time_total)

    mean_fitness_list.append(np.mean(fitness_score_RHC))
    mean_time_list.append(np.mean(time_score_RHC))
    fitness_list.append(fitness_score_RHC)


    #===================================================================================
    # =============================  G A  ==============================================
    # ==================================================================================

    fitness_score_GA = []
    time_score_GA = []

    for iter in iterations:
        start_time = time.time()
        best_state, best_fitness = mlrose.genetic_alg(problem,
                                                      max_attempts=10,
                                                      max_iters=iter,
                                                      random_state=seed,
                                                      pop_size=10,
                                                      mutation_prob=0.1)
        end_time = time.time()

        time_total = end_time - start_time
        fitness_score_GA.append(best_fitness)
        time_score_GA.append(time_total)

    mean_fitness_list.append(np.mean(fitness_score_GA))
    mean_time_list.append(np.mean(time_score_GA))
    fitness_list.append(fitness_score_GA)

    # ===================================================================================
    # =============================  S A  ==============================================
    # ==================================================================================

    fitness_score_SA = []
    time_score_SA = []

    for iter in iterations:
        start_time = time.time()
        schedule = mlrose.GeomDecay(init_temp=10, decay=0.5)
        best_state, best_fitness = mlrose.simulated_annealing(problem,
                                                              max_attempts=10,
                                                              max_iters=iter,
                                                              random_state=seed,
                                                              schedule=schedule)
        end_time = time.time()
        time_total = end_time - start_time
        fitness_score_SA.append(best_fitness)
        time_score_SA.append(time_total)


    mean_fitness_list.append(np.mean(fitness_score_SA))
    mean_time_list.append(np.mean(time_score_SA))
    fitness_list.append(fitness_score_SA)

    # ===================================================================================
    # =============================  MIMIC  ==============================================
    # ==================================================================================

    fitness_score_MIMIC = []
    time_score_MIMIC = []

    for iter in iterations:
        start_time = time.time()
        best_state, best_fitness = mlrose.mimic(problem,
                                                max_attempts=10,
                                                max_iters=iter,
                                                random_state=seed,
                                                pop_size=10)
        end_time = time.time()
        time_total = end_time - start_time
        fitness_score_MIMIC.append(best_fitness)
        time_score_MIMIC.append(time_total)



    mean_fitness_list.append(np.mean(fitness_score_MIMIC))
    mean_time_list.append(np.mean(time_score_MIMIC))
    fitness_list.append(fitness_score_MIMIC)

    # ===================================================================================
    # =============================  PLOT THE FINAL CURVE  ==============================================
    # ==================================================================================

    algorithms = ["RHC", "GA", "SA", "MIMIC"]
    algorithm_compare(fitness_list, algorithms, question, iterations)
    mean_time_compare(mean_time_list, algorithms, question)
    score_compare(mean_fitness_list, algorithms, question)

if __name__ == "__main__":
    print("TSP in")

    seed = 5
    coor = get_tsp_coords_list(seed)
    fitness_coords = mlrose.TravellingSales(coords=coor)
    problem = mlrose.TSPOpt(length=8, fitness_fn=fitness_coords, maximize=False)
    print('Created Problem')

    find_best_param_ga_pop_size(seed, problem)
    find_best_param_ga_mutation_prob(seed, problem)
    find_best_param_sa_decay(seed, problem)
    process_tsp_with_four_algo(coor, seed, problem)


    print("TSP OUT")