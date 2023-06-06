# -*- coding: UTF-8 -*-
"""
@Project:   feature-selection 
@File:      algo.py
@Author:    Rosenberg
@Date:      2022/10/7 18:47 
@Documentation: 
    This file contains the implementation of the evolutionary algorithm for feature selection.
    Mainly reference to the following articles:
        "https://www.jianshu.com/p/8fa044ed9267"
        "https://www.jianshu.com/p/3cbf5df95597"
        "https://www.jianshu.com/p/4873e16fa05a"
        "https://www.jianshu.com/p/a15d06645767"
        "https://www.jianshu.com/p/8e16fe258337"
        "https://www.jianshu.com/p/0b9da31f9ba3"
"""
import random
from typing import Union

import numpy as np
import pandas as pd
from deap import base, creator, tools
from deap.tools.emo import assignCrowdingDist
from sklearn.model_selection import cross_val_score, train_test_split
from tqdm import tqdm

import utils

creator.create("FeatureSelObj", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FeatureSelObj)  # why bother to do this?


# register functions: individual, population, evaluate, select, mate, mutate，map
def get_toolbox(
        model_name: str,
        dataset_name,
        split: Union[int, float],
        pool=None
):
    toolbox = base.Toolbox()
    toolbox.dataset_name = dataset_name
    toolbox.model_name = model_name
    toolbox.split = split

    # read dataset by name
    if type(dataset_name) is str:
        data_list = utils.dataset_paths
        dataset_path = data_list[dataset_name]
        dataset = pd.read_csv(dataset_path).values
    else:  # pass dataset directly
        dataset = dataset_name.values
        split = 0.0

    # Registers a function attr_bool with the toolbox that generates random integer values of either 0 or 1.
    # It is used for initializing the binary values in an individual.
    toolbox.register("attr_bool", random.randint, 0, 1)
    # Register a function individual with the toolbox that creates an individual for the genetic algorithm.
    # The individual is represented as a list of binary values (0 or 1), with a length equal to dataset.shape[1] - 1.
    # The tools.initRepeat function is used to initialize the individual by repeating the attr_bool function.
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        dataset.shape[1] - 1,
    )
    # The population is represented as a list of individuals.
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # The evaluate function is responsible for evaluating the fitness of individuals in the population.
    # It takes the model_name, dataset, and split as additional arguments. 
    toolbox.register("evaluate", evaluate, model_name=model_name, dataset=dataset, split=split)
    # The selection operator named selNSGA2 is used to select individuals for mating based on their fitness and dominance.
    toolbox.register("select", tools.selNSGA2)
    # The cxUniform operator performs uniform crossover on two individuals, swapping their bits with a probability of indpb.
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    # The mutFlipBit operator flips the bits of an individual with a probability of indpb.
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.3)
    # This enables parallel evaluation of individuals using the multiprocessing pool.
    if pool is not None:
        # registers the pool's map method with the toolbox. 
        toolbox.register("map", pool.map)
    # return the configured toolbox
    return toolbox


def evaluate(
        individual: np.ndarray,
        model_name: str,
        dataset: np.ndarray,
        split: Union[int, float] = 10
):
    feature_code: np.ndarray = np.asarray(individual)
    # if no feature is selected, return a worst score 0,0.
    feature_indices = np.where(feature_code == 1)[0]
    if len(feature_indices) == 0:
        return 1, 500
    # get the classifier model from sklearn package
    model = utils.models[model_name]()
    # get selected features and label is in the end of the data point by default 
    x, y = dataset[:, feature_indices], dataset[:, -1]
    # split the data to training and testing data
    if type(split) is float:
        if split == 0.0:
            model.fit(x, y)
            error_rate = 1 - model.score(x, y)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split)
            model.fit(x_train, y_train)
            error_rate = 1 - model.score(x_test, y_test)
    # Parameter split specifies the cross-validation strategy
    # It determines how the dataset will be divided into training and validation sets for evaluating the model's performance.
    elif type(split) is int:
        error_rate = 1 - np.mean(cross_val_score(model, x, y, cv=split))
    else:
        raise TypeError('split must be an integer or float.')
    # score of objective 2, count the ratio of number of features that are not selected, this one should be maximized
    n_features = np.sum(feature_code == 1)

    return error_rate, n_features


def feature_selection_with_nsga2(
        toolbox: base.Toolbox,
        num_generation: int = 256,
        num_population: int = 128,
        crossover_prob: float = 0.9,
        mutate_prob: float = 0.3,

):
    """
    Uses Genetic Algorithm to find out the best features for an input model
    using Distributed Evolutionary Algorithms in Python(DEAP) package. Default toolbox is
    used for GA, but it can be changed accordingly.
    :param toolbox: toolbox for the algorithm
    :param num_population: population size
    :param crossover_prob: crossover probability
    :param mutate_prob: mutation probability
    :param num_generation: number of generations
    :return: Fittest population
    """
    # By registering these statistical measures with the Statistics object,
    # you can compute and collect various statistics based on the fitness values of the individuals in a population. 
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # print(f"Evolving on {toolbox.dataset_name} with {toolbox.model_name} under setiing of {toolbox.split}.")

    # Generate population
    pop = toolbox.population(num_population)
    gens = [pop]  # This list can be used to store the evolution of the population over generations.

    # Initialize attributes in parallel
    fitness = toolbox.map(toolbox.evaluate, pop)  # a list of fitness score
    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit
    assignCrowdingDist(pop)  # assign a crowding distance value to each individual in the population pop.

    record = stats.compile(
        pop)  # It computes the registered statistical measures for the population and returns the results as a dictionary.
    logbook.record(gen=0, evals=len(pop), **record)  # ** syntax allows passing keyword arguments as a dictionary. 

    for _ in tqdm(  # The tqdm library is used to display a progress bar
            range(num_generation),
            bar_format='Generation {n}/{total}:|{bar}|[{elapsed}<{remaining},{rate_fmt}{postfix}]'
    ):
        # Vary the population
        # select using tournament selection with Double Tournament Dominance Comparison #？？？？
        offspring = tools.selTournamentDCD(pop, len(pop))
        # create a copy of each selected individual. Ensure that the original individuals are not modified during the variation process.
        offspring = [toolbox.clone(ind) for ind in offspring]

        # crossover
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):  # ？？？？
            if random.random() <= crossover_prob:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                del ind2.fitness.values
        # mutation
        for ind in offspring:
            if random.random() <= mutate_prob:
                toolbox.mutate(ind)
                del ind.fitness.values

        pop = utils.deduplicate(pop + offspring)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitness = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop, num_population)

        gens.append(pop)
        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)

    return gens, logbook
