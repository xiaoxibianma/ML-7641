import os
import time

import matplotlib.pyplot as plt
import mlrose
import numpy as np
from mlrose import GeomDecay
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier

from ut import pruning_titanic_dataset


# https://towardsdatascience.com/fitting-a-neural-network-using-randomized-optimization-in-python-71595de4ad2d
def get_nn_error(x_train_scaled, x_test_scaled, y_train, y_test, iter, param_name, param_value, algorithm):
    # Initialize neural network object and fit object
    start_time = time.time()



    # Random Hill Climb
    if param_name == "":
        nn_ro = mlrose.NeuralNetwork(hidden_nodes=[5], activation='relu', algorithm=algorithm,
                                  max_iters=iter, bias=True, is_classifier=True, learning_rate=0.0001,
                                  early_stopping=True, clip_max=5, max_attempts=100,
                                  random_state=3)
    # Simulated Annealing
    elif param_name == "init_temp":
        nn_ro = mlrose.NeuralNetwork(hidden_nodes=[5], activation='relu', algorithm='simulated_annealing',
                                      max_iters=iter, bias=True, is_classifier=True, learning_rate=0.0001,
                                      early_stopping=True, clip_max=5, max_attempts=100,
                                      random_state=3, schedule=GeomDecay(init_temp=param_value))
    # Simulated Annealing
    elif param_name == "decay":
        nn_ro = mlrose.NeuralNetwork(hidden_nodes=[5], activation='relu', algorithm='simulated_annealing',
                                      max_iters=iter, bias=True, is_classifier=True, learning_rate=0.0001,
                                      early_stopping=True, clip_max=5, max_attempts=100,
                                      random_state=3, schedule=GeomDecay(init_temp=param_value))

    elif param_name == "pop_size":
        nn_ro = mlrose.NeuralNetwork(hidden_nodes=[5], activation='relu', algorithm='genetic_alg',
                                     max_iters=iter, bias=True, is_classifier=True, learning_rate=0.0001,
                                     early_stopping=True, clip_max=5, max_attempts=100,
                                     pop_size=param_value,
                                     random_state=3)


    elif param_name == "mutation_prob":
        nn_ro = mlrose.NeuralNetwork(hidden_nodes=[5], activation='relu', algorithm='genetic_alg',
                                     max_iters=iter, bias=True, is_classifier=True, learning_rate=0.0001,
                                     early_stopping=True, clip_max=5, max_attempts=100,
                                     mutation_prob=param_value,
                                     random_state=3)

    nn_ro.fit(x_train_scaled, y_train)
    # get running time
    end_time = time.time()
    train_time = end_time - start_time

    # get prediction accuracy
    y_pred = nn_ro.predict(x_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Get Mean Squared Error
    MSE = mean_squared_error(y_test, y_pred)
    return test_accuracy, MSE, train_time


def random_algo_train(x_train_scaled, x_test_scaled, y_train, y_test, iters, param_name, param_value, algorithm):
    print(algorithm)
    times = []
    accuracies = []
    errors = []
    for iter in iters:
        accuracy, MSE, train_time = get_nn_error(
            x_train_scaled, x_test_scaled, y_train, y_test, iter, param_name, param_value, algorithm)

        times.append(train_time)
        accuracies.append(accuracy)
        errors.append(MSE)

    mean_time = np.mean(times)
    max_accuracy = max(accuracies)
    min_error = min(errors)

    return errors, accuracies, max_accuracy, min_error, mean_time


def get_bp_error(x_train, x_test, y_train, y_test, iter):
    start_time = time.time()
    clf = MLPClassifier(random_state=10, max_iter=iter, hidden_layer_sizes=5, activation='relu')
    train_sizes, train_scores, test_scores = learning_curve(
        clf, x_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 20))
    clf.fit(x_train, y_train)

    # get training time
    end_time = time.time()
    train_time = end_time - start_time
    # get prediction time
    y_predict = clf.predict(x_test)
    pred_time = time.time() - end_time

    # get prediction accuracy
    y_pred = clf.predict(x_test)
    pred_accuracy = accuracy_score(y_test, y_pred)
    # Get Mean Squared Error
    MSE = mean_squared_error(y_test, y_pred)

    return pred_accuracy, MSE, train_time, pred_time


def nn_back_propagation(x_train, x_test, y_train, y_test, iters):
    times = []
    accuracies = []
    errors = []
    for iter in iters:
        accuracy, MSE, train_time, pred_time = get_bp_error(x_train, x_test, y_train, y_test, iter)
        times.append(train_time)
        accuracies.append(accuracy)
        errors.append(MSE)

    mean_time = np.mean(times)
    max_accuracy = max(accuracies)
    min_error = min(errors)

    return errors, accuracies, max_accuracy, min_error, mean_time


def plot_nn_algorithm_compare(accuracies, algorithms, iters):
    x_axis = iters

    plt.close()
    # Create plot
    plt.figure()
    plt.title("Neural Network Weight Optimization Algorithms Comparison")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")

    # Draw lines
    plt.grid()
    colors = ['r', 'b', 'g', 'm']
    idx = 0
    for accuracy in accuracies:
        plt.plot(x_axis, accuracy, label=algorithms[idx], color=colors[idx])
        idx += 1
    plt.legend(loc="best")
    plt.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('.', "NN Weight Algorithm Compare")
    plt.savefig(image_path, dpi=100)
    return

if __name__ == "__main__":
    data, x_train, x_train_scaled, x_test, x_test_scaled, y_train, y_test = pruning_titanic_dataset()

    params = {"init_temp": [0.1, 10, 100], "decay": [0.1, 0.5, 1.0],
              "pop_size": [50, 200, 500], "mutation_prob": [0.4, 0.7, 1.0]}


    # compare RHC, SA, GA
    optimized_params = {"init_temp": 1, "mutation_prob": 0.4}
    algorithms = ["back_propagation", "random_hill_climb", "simulated_annealing", "genetic_alg"]


    new_iters = [1, 3, 5, 7, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300, 500, 800,
                 1000, 1500, 2000, 3000, 5000, 6000, 8000, 10000]

    accuracies = []
    for algorithm in algorithms:
        if algorithm == "back_propagation":
            errors, accuracy, max_accuracy, min_error, mean_time = nn_back_propagation(
                x_train, x_test, y_train, y_test, new_iters)
        if algorithm == "random_hill_climb":
            errors, accuracy, max_accuracy, min_error, mean_time = random_algo_train(
                x_train_scaled, x_test_scaled, y_train, y_test, new_iters, "", 0, algorithm)
        elif algorithm == "simulated_annealing":
            errors, accuracy, max_accuracy, min_error, mean_time = random_algo_train(
                x_train_scaled, x_test_scaled, y_train, y_test, new_iters, "init_temp", 1, algorithm)
        else:
            errors, accuracy, max_accuracy, min_error, mean_time = random_algo_train(
                x_train_scaled, x_test_scaled, y_train, y_test, new_iters, "mutation_prob", 0.4, algorithm)



        accuracies.append(accuracy)


    plot_nn_algorithm_compare(accuracies, algorithms, new_iters)