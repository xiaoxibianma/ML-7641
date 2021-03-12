import matplotlib.pyplot as plt
import time
import os
import numpy as np
import pandas as pd
import re
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def ga_parameter_curve(xlist, fitness_list, parameter, width, pb):
    x_axis = xlist
    plt.close()

    plt.figure()
    plt.title("GA Algorithm Optimized Parameterfor {} (tyang358)".format(parameter))
    plt.xlabel(parameter + " ")
    plt.ylabel("Mean of Fitness Score")

    plt.bar(x_axis, fitness_list, color='b', width=width)
    plt.legend(loc="best")
    plt.tight_layout()

    plt.draw()
    image_path = os.path.join('.', "GA Algorithm Optimized Parameter for {} in {} (tyang358)".format(parameter, pb))
    plt.savefig(image_path, dpi=100)
    return


def sa_parameter_curve(xlist, fitness_list, parameter, width, pb):
    x_axis = xlist
    plt.close()

    plt.figure()
    plt.title("SA Algorithm Optimized Parameterfor {} (tyang358)".format(parameter))
    plt.xlabel(parameter + " ")
    plt.ylabel("Mean of Fitness Score")

    plt.bar(x_axis, fitness_list, color='b', width=width)
    plt.legend(loc="best")
    plt.tight_layout()

    plt.draw()
    image_path = os.path.join('.', " SA Algorithm Optimized Parameter for {} in {} (tyang358)".format(parameter, pb))
    plt.savefig(image_path, dpi=100)
    return

def plot_nn_ro_curves(errors, algorithm, param_name, param_values, iters):
    x_axis = iters

    plt.close()
    # Create plot
    plt.figure()
    plt.title("Neural Network Weight Optimization Using {} Algorithm - Impact of {}".format(algorithm, param_name), fontsize=9)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error")

    # Draw lines
    plt.grid()
    colors = ['r', 'b', 'g']
    idx = 0
    for error in errors:
        plt.plot(x_axis, error, label=param_name + "_" + str(param_values[idx]), color=colors[idx])
        idx += 1
    plt.legend(loc="best")
    plt.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('./images', "NN Weight Optimization Using {}_{}".format(algorithm, param_name))
    plt.savefig(image_path, dpi=100)
    return


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
    image_path = os.path.join('./images', "NN Weight Algorithm Compare")
    plt.savefig(image_path, dpi=100)
    return


def plot_randomized_optimization_curve(question, fitness_list, algorithm, params_names, params_values, iters):
    x_axis = iters

    plt.close()
    # Create plot
    plt.figure()
    plt.title("{} {} Optimization - Impact of {} & {}".format(question, algorithm, params_names[0], params_names[1]),
              fontsize=8)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")

    # Draw lines
    plt.grid()
    colors = ['r', 'b', 'g']
    color_idx = 0
    param_idx = 0


    plt.legend(loc="best")
    plt.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('.', "{} {} Optimization".format(question, algorithm))
    plt.savefig(image_path, dpi=100)
    return


def algorithm_compare(fitness_list, algorithms, question, iters):
    x_axis = iters

    plt.close()
    # Create plot
    plt.figure()
    plt.title("Randomized Optimization Algorithms Comparison for {} Problem (tyang358)".format(question), fontsize=9)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")

    # Draw lines
    plt.grid()
    colors = ['r', 'b', 'g', 'm']
    idx = 0
    for fitness in fitness_list:
        plt.plot(x_axis, fitness, label=algorithms[idx], color=colors[idx])
        idx += 1
    plt.legend(loc="best")
    plt.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('.', "{} Algorithm Compare".format(question))
    plt.savefig(image_path, dpi=100)
    return

def mean_time_compare(mean_time_list, algorithms, question):
    x_axis = algorithms

    plt.close()
    # Create plot
    plt.figure()
    plt.title("Algorithms Running Time Comparison for {} Problem (tyang358)".format(question), fontsize=9)
    plt.xlabel("Algorithm")
    plt.ylabel("Training Time/log")

    # Draw lines
    # plt.grid()
    # creating the bar plot
    plt.bar(x_axis, mean_time_list, color='b', width=0.4)
    plt.yscale('log')
    plt.legend(loc="best")
    plt.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('.', "{} Time Compare".format(question))
    plt.savefig(image_path, dpi=100)
    return

def score_compare(converged_fitness_list, algorithms, question):
    x_axis = algorithms

    plt.close()
    # Create plot
    plt.figure()
    plt.title("Algorithms Mean Fitness Comparison for {} Problem (tyang358)".format(question), fontsize=9)
    plt.xlabel("Algorithm")
    plt.ylabel("Converged Fitness")

    # Draw lines
    # plt.grid()
    # creating the bar plot
    plt.bar(x_axis, converged_fitness_list, color='g', width=0.4)
    plt.legend(loc="best")
    plt.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('.', "{} Fitness Compare".format(question))
    plt.savefig(image_path, dpi=100)
    return

def pruning_titanic_dataset():
    filename = './titanic.csv'
    data = pd.read_csv(filename)
    # Feature that tells whether a passenger had a cabin on the Titanic
    data['Has_Cabin'] = data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    # %% [code]
    # Create new feature FamilySize as a combination of SibSp and Parch
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    # Create new feature IsAlone from FamilySize
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

    # Remove all NULLS in the Embarked column
    data['Embarked'] = data['Embarked'].fillna('S')

    # Remove all NULLS in the Fare column
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

    # Remove all NULLS in the Age column
    age_avg = data['Age'].mean()
    age_std = data['Age'].std()
    age_null_count = data['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    # Next line has been improved to avoid warning
    data.loc[np.isnan(data['Age']), 'Age'] = age_null_random_list
    data['Age'] = data['Age'].astype(int)

    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""

    data['Title'] = data['Name'].apply(get_title)
    # Group all non-common titles into one single grouping "Rare"
    data['Title'] = data['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    # Mapping Sex
    # Remove all NULLS in the Sex column
    data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # Mapping titles
    title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)

    # Mapping Embarked
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # Mapping Fare
    data.loc[data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2
    data.loc[data['Fare'] > 31, 'Fare'] = 3
    data['Fare'] = data['Fare'].astype(int)

    # Mapping Age
    data.loc[data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[data['Age'] > 64, 'Age']

    # Feature selection: remove variables no longer containing relevant information
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
    data = data.drop(drop_elements, axis=1)
    X = data.drop(['Survived'], axis=1)
    y = data['Survived']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return data, x_train, x_train_scaled, x_test, x_test_scaled, y_train, y_test
