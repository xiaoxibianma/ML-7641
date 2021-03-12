CS 7641
Tianyi Yang
tyang358
Assignment 2

TSP_prob.py : the file helps to create a Travelling Salesman problem, and optimized different parameters for SA, GA, and MIMIC
              Then, the function "process_tsp_with_four_algo(coor, seed, problem)" is used for compared RHC, SA, GA and MIMIC
              for same problems.

FP.py:  the file first generate a Four(Continuous) Peaks Problem, and optimized different parameters for SA, GA, and MIMIC
              Then, the function "process_cp_with_four_algo(seed, problem)" is used for compared RHC, SA, GA and MIMIC
              for same problems.

Knap.py: the file first generate a 0-1(Knapsack) problem, and then optimized different parameters for SA, GA, and MIMIC
              finally, the function "process_cp_with_four_algo(seed, problem)" is used for compared RHC, SA, GA and MIMIC
              algorithms in one Knapsack problem.

optimized_nn.py: the file first tunning the Titanic dataset, then optimized different parameters for SA, GA, and MIMIC
              finally, the function use the same dataset to compared RHC, SA, GA and PB algorithm in terms of observe their
              MSE the time cost

ut.py: contains some functions to plot the chart and pre-prunning the Titanic dataset.


