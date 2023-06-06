import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import tools
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from model import feature_selection_with_nsga2, get_toolbox
import utils

path = './'
hv = 0


def preprocess_dataset(filename, col_headings, column_name_class='', column_list_drop=[]):
    df = pd.read_csv(path + filename, na_filter=True)
    if len(col_headings) != 0:
        df.columns = col_headings

    if column_name_class != '':
        class_column = df.pop(column_name_class)
        df['class'] = class_column

    if len(column_list_drop) != 0:
        for _ in column_list_drop:
            df = df.drop(_, axis=1)

    if filename == 'sonar.data':
        df.loc[df[column_name_class] == 'M', column_name_class] = 1
        df.loc[df[column_name_class] == 'B', column_name_class] = 0

    df.to_csv(path + filename[:filename.find('.')] + '.csv', index=False)


def main(i, file_name):
    name = file_name.split(".")[0]
    features = ['V' + str(i) for i in range(1, 167)]

    setting = 0.0
    split = 0.3
    model_name = '5NN'
    df = pd.read_csv(
        path + file_name[:file_name.find('.')] + '.csv', na_filter=True)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=split)
    training_data = x_train.join(y_train)
    knn5_train = utils.models[model_name]()
    knn5_test = utils.models[model_name]()
    knn5_train.fit(x_train, y_train)
    knn5_test.fit(x_test, y_test)
    y_pred_train = knn5_train.predict(x_train)
    y_pred_test = knn5_test.predict(x_test)
    error_all_train = 1 - accuracy_score(y_train, y_pred_train)
    error_all_test = 1 - accuracy_score(y_test, y_pred_test)

    with multiprocessing.Pool() as pool:
        toolbox = get_toolbox(model_name, training_data, setting, pool)
        gens, _ = feature_selection_with_nsga2(toolbox,
                                               num_generation=100,
                                               num_population=100,
                                               crossover_prob=0.9,
                                               mutate_prob=0.01)
    fronts_first = tools.sortNondominated(gens[0], len(gens[0]))
    fronts_last = tools.sortNondominated(gens[-1], len(gens[-1]))
    pareto_front_first = np.asarray(
        [ind.fitness.values for ind in fronts_first[0]])
    pareto_front_last = np.asarray(
        [ind.fitness.values for ind in fronts_last[0]])

    error_rate_first = list(map(lambda x: x[0], pareto_front_first))
    n_features_first = list(map(lambda x: x[1], pareto_front_first))
    error_rate_last = list(map(lambda x: x[0], pareto_front_last))
    n_features_last = list(map(lambda x: x[1], pareto_front_last))
    pop = gens[-1]  # population of last generation
    sorted_pop_last = utils.sort_population(pop)
    best_solution = sorted_pop_last[0]
    best_slt_code = np.asarray(best_solution)
    feature_indices = np.where(best_slt_code == 1)[0]
    x_test2 = x_test.values[:, feature_indices]
    knn5 = utils.models[model_name]()
    knn5.fit(x_test2, y_test)
    y_pred_test2 = knn5.predict(x_test2)
    error_all_test2 = 1 - accuracy_score(y_test, y_pred_test2)
    hv_local = tools.hypervolume(pop, ref_point=(-1.0, -1.0))

    global hv
    hv += hv_local
    print("Error rate on train set by using all features: {:.2%}".format(
        error_all_train))
    print("Error rate on test set by using all features: {:.2%}".format(
        error_all_test))
    print(f'Indices of the best solution: {feature_indices}')
    print('Number of Features: {} out of {}'.format(
        best_solution.fitness.values[1], len(features)))
    print("Error Rate on training set: {:.2%}".format(
        best_solution.fitness.values[0]))
    print("Error Rate on test set    : {:.2%}".format(error_all_test2))
    print(f"HV: {hv_local}\n")

    plt.figure()
    plt.title(f"{name} Run: {i}")
    plt.scatter(n_features_first, error_rate_first,
                c="red", marker="^", label="Initial")
    plt.scatter(n_features_last, error_rate_last,
                c="orange", marker=".", label="Final")
    plt.scatter(sorted_pop_last[0].fitness.values[1],
                sorted_pop_last[0].fitness.values[0], c="green", marker="o", label="Optimum")
    plt.xlabel('Features')
    plt.ylabel('Classification Error')
    pic_name = f"{i}_{name}"
    plt.legend()
    plt.savefig(f"images/{pic_name}", dpi=600)


if __name__ == "__main__":
    print("Make your selection:")
    print("     1. Low-scale (Sonar)")
    print("     2. Large-scale (Arrhythmia)")
    ch = int(input("Enter: "))

    file_name = "sonar.csv" if ch == 1 else "arrhythmia.csv"

    for i in range(15):
        print(f"Run {i+1}")
        main(i + 1, file_name)
    print(f"Average Hypervolume over 15 runs: {hv/15}\n")
