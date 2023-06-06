import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import tools
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from model import feature_selection_with_nsga2, get_toolbox
import utils

path = './'  # the path of input and output data files
hv = 0
# matplotlib.use('Agg')


def preprocess_dataset(filename, col_headings, column_name_class='', column_list_drop=[]):
    # preprocess the dataset
    df = pd.read_csv(path + filename, na_filter=True)
    if len(col_headings) != 0:
        df.columns = col_headings  # add headings

    if column_name_class != '':
        # Remove the column from the DataFrame and store it
        class_column = df.pop(column_name_class)
        # Insert the column at the end of the dataframe column-wise
        df['class'] = class_column
        # df.insert(new_position, column_name, column)

    # drop the undesired column
    if len(column_list_drop) != 0:
        for _ in column_list_drop:
            df = df.drop(_, axis=1)

    # Assign 1 to the string 'M' in column 'M'
    if filename == 'sonar.data':
        df.loc[df[column_name_class] == 'M', column_name_class] = 1
        df.loc[df[column_name_class] == 'B', column_name_class] = 0

    df.to_csv(path + filename[:filename.find('.')] + '.csv', index=False)
    # print(df.head)
    # print('Size of dataset:', filename, df.shape)


def main(i):
    file_name = 'arrhythmia.csv'
    features = ['V' + str(i) for i in range(1, 167)]

    # NSGA-II
    setting = 0.0  # Use all training data in NSGA-2
    split = 0.3  # 30% testing data
    model_name = '5NN'
    dataset_name = file_name[:file_name.find('.')]
    df = pd.read_csv(
        path + file_name[:file_name.find('.')] + '.csv', na_filter=True)
    # Separate the features (X) and the target variable (y)
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target variable
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=split)
    training_data = x_train.join(y_train)
    # get the error rate of using all features
    knn5_train = utils.models[model_name]()
    knn5_test = utils.models[model_name]()
    knn5_train.fit(x_train, y_train)
    knn5_test.fit(x_test, y_test)
    # Use the trained classifier to make predictions on the train set
    y_pred_train = knn5_train.predict(x_train)
    # Use the trained classifier to make predictions on the test set
    y_pred_test = knn5_test.predict(x_test)
    # Calculate error on train set
    error_all_train = 1 - accuracy_score(y_train, y_pred_train)
    # Calculate error on test set
    error_all_test = 1 - accuracy_score(y_test, y_pred_test)

    with multiprocessing.Pool() as pool:
        toolbox = get_toolbox(model_name, training_data, setting, pool)
        gens, _ = feature_selection_with_nsga2(toolbox,
                                               num_generation=100,
                                               num_population=100,
                                               crossover_prob=0.9,
                                               mutate_prob=0.01)
    # Sort the individuals based on Pareto dominance
    fronts_first = tools.sortNondominated(gens[0], len(gens[0]))
    fronts_last = tools.sortNondominated(gens[-1], len(gens[-1]))
    # get the pareto front of fist rank
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
    # calculate the error on test set by using features of the best solution
    feature_indices = np.where(best_slt_code == 1)[0]
    x_test2 = x_test.values[:, feature_indices]
    knn5 = utils.models[model_name]()
    knn5.fit(x_test2, y_test)
    # Use the trained classifier to make predictions on the test set
    y_pred_test2 = knn5.predict(x_test2)
    # Calculate error on test set
    error_all_test2 = 1 - accuracy_score(y_test, y_pred_test2)
    # calculate HV
    # hv_pareto_front = [individual for individual in pop]
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

    plt.subplot(5, 3, i)
    plt.title(f"{file_name} Run: {i}")
    plt.scatter(n_features_first, error_rate_first, c="purple", marker="^")
    plt.scatter(n_features_last, error_rate_last, c="blue", marker="o")
    plt.scatter(sorted_pop_last[0].fitness.values[1],
                sorted_pop_last[0].fitness.values[0], c="r")
    plt.xlabel('Features')
    plt.ylabel('Classification Error')
    # plt.show()
    # current_time = datetime.datetime.now()  # Get the current time
    # Format the current time as a string
    # time_string = current_time.strftime("%Y%m%d%H%M%S")
    # pic_name = f"{path}images/{time_string[-4:]}_{file_name}.png"
    # plt.savefig(pic_name, dpi=600)


if __name__ == "__main__":
    plt.subplots(5, 3)
    for i in range(15):
        print(f"Run {i+1}")
        main(i + 1)
    plt.show()
    print(f"Average Hypervolume over 15 runs: {hv/15}\n")
