import random
import warnings
from itertools import combinations, product, islice, chain
from math import log, ceil
from multiprocessing.pool import Pool
import pandas as pd
import numpy as np
from pandas import DataFrame, merge
from scipy.optimize import fsolve

from DataSynthesizer.lib.utils import mutual_information, normalize_given_distribution, set_random_seed

"""
This module is based on PrivBayes in the following paper:

Zhang J, Cormode G, Procopiuc CM, Srivastava D, Xiao X.
PrivBayes: Private Data Release via Bayesian Networks.
"""


def calculate_sensitivity(num_tuples, child, parents, attr_to_is_binary):
    """Sensitivity function for Bayesian network construction. PrivBayes Lemma 1.

    Parameters
    ----------
    num_tuples : int
        Number of tuples in sensitive dataset.

    Return
    --------
    int
        Sensitivity value.
    """
    if attr_to_is_binary[child] or (len(parents) == 1 and attr_to_is_binary[parents[0]]):
        a = log(num_tuples) / num_tuples
        b = (num_tuples - 1) / num_tuples
        b_inv = num_tuples / (num_tuples - 1)
        return a + b * log(b_inv)
    else:
        a = (2 / num_tuples) * log((num_tuples + 1) / 2)
        b = (1 - 1 / num_tuples) * log(1 + 2 / (num_tuples - 1))
        return a + b


def calculate_delta(num_attributes, sensitivity, epsilon):
    """Computing delta, which is a factor when applying differential privacy.

    More info is in PrivBayes Section 4.2 "A First-Cut Solution".

    Parameters
    ----------
    num_attributes : int
        Number of attributes in dataset.
    sensitivity : float
        Sensitivity of removing one tuple.
    epsilon : float
        Parameter of differential privacy.
    """
    return (num_attributes - 1) * sensitivity / epsilon


def usefulness_minus_target(k, num_attributes, num_tuples, target_usefulness=5, epsilon=0.1):
    """Usefulness function in PrivBayes.

    Parameters
    ----------
    k : int
        Max number of degree in Bayesian networks construction
    num_attributes : int
        Number of attributes in dataset.
    num_tuples : int
        Number of tuples in dataset.
    target_usefulness : int or float
    epsilon : float
        Parameter of differential privacy.
    """
    if k == num_attributes:
        usefulness = target_usefulness
    else:
        usefulness = num_tuples * epsilon / ((num_attributes - k) * (2 ** (k + 3)))  # PrivBayes Lemma 3
    return usefulness - target_usefulness


def calculate_k(num_attributes, num_tuples, target_usefulness=4, epsilon=0.1):
    """Calculate the maximum degree when constructing Bayesian networks. See PrivBayes Lemma 3."""
    default_k = 3
    initial_usefulness = usefulness_minus_target(default_k, num_attributes, num_tuples, 0, epsilon)
    if initial_usefulness > target_usefulness:
        return default_k
    else:
        arguments = (num_attributes, num_tuples, target_usefulness, epsilon)
        warnings.filterwarnings("error")
        try:
            ans = fsolve(usefulness_minus_target, np.array([int(num_attributes / 2)]), args=arguments)[0]
            ans = ceil(ans)
        except RuntimeWarning:
            print("Warning: k is not properly computed!")
            ans = default_k
        if ans < 1 or ans > num_attributes:
            ans = default_k
        return ans


def worker(paras):
    child, V, num_parents, split, dataset = paras
    parents_pair_list = []
    mutual_info_list = []

    if split + num_parents - 1 < len(V):
        for other_parents in combinations(V[split + 1:], num_parents - 1):
            parents = list(other_parents)
            parents.append(V[split])
            parents_pair_list.append((child, parents))
            # TODO consider to change the computation of MI by combined integers instead of strings.
            mi = mutual_information(dataset[child], dataset[parents])
            mutual_info_list.append(mi)

    return parents_pair_list, mutual_info_list


def greedy_bayes(dataset: DataFrame, k: int, epsilon: float, seed=0):
    """Construct a Bayesian Network (BN) using greedy algorithm.

    Parameters
    ----------
    dataset : DataFrame
        Input dataset, which only contains categorical attributes.
    k : int
        Maximum degree of the constructed BN. If k=0, k is automatically calculated.
    epsilon : float
        Parameter of differential privacy.
    seed : int or float
        Seed for the randomness in BN generation.
    """
    set_random_seed(seed)
    dataset: DataFrame = dataset.astype(str, copy=False)
    num_tuples, num_attributes = dataset.shape
    if not k:
        k = calculate_k(num_attributes, num_tuples)

    attr_to_is_binary = {attr: dataset[attr].unique().size <= 2 for attr in dataset}

    print('================ Constructing Bayesian Network (BN) ================')
    root_attribute = random.choice(dataset.columns)
    V = [root_attribute]
    rest_attributes = list(dataset.columns)
    rest_attributes.remove(root_attribute)
    print(f'Adding ROOT {root_attribute}')
    N = []
    while rest_attributes:
        parents_pair_list = []
        mutual_info_list = []

        num_parents = min(len(V), k)
        tasks = [(child, V, num_parents, split, dataset) for child, split in
                 product(rest_attributes, range(len(V) - num_parents + 1))]
        with Pool() as pool:
            res_list = pool.map(worker, tasks)

        for res in res_list:
            parents_pair_list += res[0]
            mutual_info_list += res[1]

        if epsilon:
            sampling_distribution = exponential_mechanism(epsilon, mutual_info_list, parents_pair_list, attr_to_is_binary,
                                                          num_tuples, num_attributes)
            idx = np.random.choice(list(range(len(mutual_info_list))), p=sampling_distribution)
        else:
            idx = mutual_info_list.index(max(mutual_info_list))

        N.append(parents_pair_list[idx])
        adding_attribute = parents_pair_list[idx][0]
        V.append(adding_attribute)
        rest_attributes.remove(adding_attribute)
        print(f'Adding attribute {adding_attribute}')

    print('========================== BN constructed ==========================')

    return N


def exponential_mechanism(epsilon, mutual_info_list, parents_pair_list, attr_to_is_binary, num_tuples, num_attributes):
    """Applied in Exponential Mechanism to sample outcomes."""
    delta_array = []
    for (child, parents) in parents_pair_list:
        sensitivity = calculate_sensitivity(num_tuples, child, parents, attr_to_is_binary)
        delta = calculate_delta(num_attributes, sensitivity, epsilon)
        delta_array.append(delta)

    mi_array = np.array(mutual_info_list) / (2 * np.array(delta_array))
    mi_array = np.exp(mi_array)
    mi_array = normalize_given_distribution(mi_array)
    return mi_array


def laplace_noise_parameter(k, num_attributes, num_tuples, epsilon):
    """The noises injected into conditional distributions.

    Note that these noises are over counts, instead of the probability distributions in PrivBayes Algorithm 1.
    """
    return (num_attributes - k) / epsilon


def get_noisy_distribution_of_attributes(attributes, encoded_dataset, epsilon=0.1, partial_noise=False, number_of_clients=None):
    data = encoded_dataset.copy(deep=True).loc[:, attributes]
    data['count'] = 1
    stats = data.groupby(attributes,observed=False).sum() # Changed to accept all possible combinations even if not present
    try: 
        print('try iterables')
        iterables = [range(len(data[attr].cat.categories.unique())) for attr in attributes] # Added to accept all possible combinations even if not present 
    except:
        print('no try iterables')
        iterables = [range(int(encoded_dataset[attr].max()) + 1) for attr in attributes] 
    
    products = product(*iterables)

    def grouper_it(iterable, n):
        while True:
            chunk_it = islice(iterable, n)
            try:
                first_el = next(chunk_it)
            except StopIteration:
                return
            yield chain((first_el,), chunk_it)

    full_space = None
    for item in grouper_it(products, 1000000):
        if full_space is None:
            full_space = DataFrame(columns=attributes, data=list(item))
        else:
            data_frame_append = DataFrame(columns=attributes, data=list(item))
            full_space = pd.concat([full_space, data_frame_append], ignore_index=True)

    stats.reset_index(inplace=True)
    stats = merge(full_space, stats, how='left')
    stats.fillna(0, inplace=True)
    
    if epsilon:
        k = len(attributes) - 1
        num_tuples, num_attributes = encoded_dataset.shape
        noise_para = laplace_noise_parameter(k, num_attributes, num_tuples, epsilon)
        if partial_noise: # Added for FedEnhaced DataSynthesizer
            size=stats.index.size
            scale = noise_para/(2*number_of_clients)
            partial_noises=np.random.normal(loc=0,scale=scale,size=size)**2+np.random.normal(loc=0,scale=scale,size=size)**2-np.random.normal(loc=0,scale=scale,size=size)**2-np.random.normal(loc=0,scale=scale,size=size)**2
            stats['count'] += partial_noises
            stats.loc[stats['count'] < 0, 'count'] = 0
        else:
            laplace_noises = np.random.laplace(0, scale=noise_para, size=stats.index.size)
            stats['count'] += laplace_noises
            stats.loc[stats['count'] < 0, 'count'] = 0

    return stats

def get_noisy_distribution_kplus1_attributes(bayesian_network,encoded_dataset,epsilon,partial_noise,number_of_clients):
    # Added for FedEnhancedDataSynthesizer
    k = len(bayesian_network[-1][1])

    # first k+1 attributes
    root = bayesian_network[0][1][0]
    kplus1_attributes = [root]
    for child, _ in bayesian_network[:k]:
        kplus1_attributes.append(child)

    noisy_dist_of_kplus1_attributes = get_noisy_distribution_of_attributes(kplus1_attributes, encoded_dataset, epsilon,partial_noise,number_of_clients)
    return noisy_dist_of_kplus1_attributes

def get_all_noisy_frequencies(bayesian_network,encoded_dataset,epsilon,partial_noise,number_of_clients):
    # Added for FedEnhancedDataSynthesizer
    k = len(bayesian_network[-1][1])
    noisy_frequencies=[]
    noisy_k1 = get_noisy_distribution_kplus1_attributes(bayesian_network,encoded_dataset,epsilon,partial_noise,number_of_clients)
    noisy_frequencies.append(np.array(noisy_k1.values.tolist()))
    for idx, (child, parents) in enumerate(bayesian_network):
        if idx >= k:
            stats = get_noisy_distribution_of_attributes(parents + [child], encoded_dataset, epsilon, partial_noise, number_of_clients)
            noisy_frequencies.append(np.array(stats.values.tolist()))
    return noisy_frequencies

def get_modify_frequencies_with_threshold(bayesian_network,encoded_dataset,threshold, method='suppression',noise_range=10, base=5):
    # Added for FedEnhancedDataSynthesizer
    k = len(bayesian_network[-1][1])
    frequencies = get_all_noisy_frequencies(bayesian_network,encoded_dataset,epsilon=0)
    for table in frequencies:
        if method == 'suppression':
            # Identify the indices where count values are below the threshold
            row_indices = np.where((table[:, k+1] > 0) & (table[:, k+1] < threshold))[0]
            # Apply the function (suppression) to the values within the specified range in column 3
            table[row_indices, k+1] = np.vectorize(lambda x: x-x)(table[row_indices, k+1])
        elif method == 'perturbation':
            # Identify the indices where count values are below the threshold
            row_indices = np.where((table[:, k+1] > 0) & (table[:, k+1] < threshold))[0]
            # Apply the function (additive perturbation) to the values within the specified range in column 3
            table[row_indices, k+1] = np.vectorize(lambda x: 0 if x+np.random.randint(-noise_range, noise_range + 1) < 0 else x+np.random.randint(-noise_range, noise_range + 1))(table[row_indices, k+1])
        else:
            base=base
            table[:, k+1] = np.round(table[:, k+1] / base) * base
    return frequencies
    
    

def construct_noisy_conditional_distributions(bayesian_network, encoded_dataset, epsilon=0.1):
    """See more in Algorithm 1 in PrivBayes."""

    k = len(bayesian_network[-1][1])
    conditional_distributions = {}

    # first k+1 attributes
    root = bayesian_network[0][1][0]
    kplus1_attributes = [root]
    for child, _ in bayesian_network[:k]:
        kplus1_attributes.append(child)

    noisy_dist_of_kplus1_attributes = get_noisy_distribution_of_attributes(kplus1_attributes, encoded_dataset, epsilon)

    # generate noisy distribution of root attribute.
    root_stats = noisy_dist_of_kplus1_attributes.loc[:, [root, 'count']].groupby(root).sum()['count']
    conditional_distributions[root] = normalize_given_distribution(root_stats).tolist()

    for idx, (child, parents) in enumerate(bayesian_network):
        conditional_distributions[child] = {}

        if idx < k:
            stats = noisy_dist_of_kplus1_attributes.copy().loc[:, parents + [child, 'count']]
            stats = stats.groupby(parents + [child], as_index=False,observed=False).sum()
        else:
            stats = get_noisy_distribution_of_attributes(parents + [child], encoded_dataset, epsilon)
            stats = stats.loc[:, parents + [child, 'count']]

        for parents_instance, stats_sub in stats.groupby(parents):
            stats_sub = stats_sub.sort_values(by=child)
            dist = normalize_given_distribution(stats_sub['count']).tolist()

            if len(parents) == 1:
                parents_key = str([parents_instance])
            else:
                parents_key = str(list(parents_instance))

            conditional_distributions[child][parents_key] = dist

    return conditional_distributions
