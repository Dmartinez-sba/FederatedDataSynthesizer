from typing import Dict, List
import numpy as np
import pandas as pd
from DataSynthesizer.lib.utils import aggregate_counts, convert_individual_to_network, nested_dict_keys,deep_get_from_dict,nested_set, normalize_given_distribution
from decimal import Decimal,getcontext

class FedServer:
    def __init__(self, n_clients: int, metadata: Dict = {}):
        self.number_of_clients = n_clients
        self.attributes_metadata = metadata
        self.params_aggregation = []
        self.server_round = None 
        self.global_bayesian_network = None
        self.conditional_probabilities = {}
        
    def get_attributes_metadata(self, metadata: Dict):
        ''' The server asks one of the clients in the federated process for the attributes names and the types'''
        self.attributes_metadata = metadata
        self.attributes_metadata['attributes_description'] = {}
        
    def compute_global_min_max_numeric_columns(self, min_max_clients: List):
        ''' The server computes the global maximum and global minimum to discretize numeric values'''
        for attribute in self.attributes_metadata['numerical_attributes']:
            global_min = min(list(map(lambda d: d[attribute]['min'],  min_max_clients)))
            global_max = max(list(map(lambda d: d[attribute]['max'],  min_max_clients)))
            distribution_bins = [global_min,global_max]
            self.attributes_metadata['attributes_description'][attribute] = {'min': global_min, 'max': global_max, 'distribution_bins':distribution_bins}
        
    def align_categorical_columns(self, categories_clients: List):
        ''' The server aligns the categorical columns of the different clients '''
        for attribute in self.attributes_metadata['categorical_attributes']:
            categories_per_attribute = list(map(lambda d: d[attribute]['categories'],  categories_clients))
            all_categories_per_attribute=set().union(*categories_per_attribute)
            self.attributes_metadata['attributes_description'][attribute] = {'distribution_bins': list(all_categories_per_attribute)}
            
    def aggregate_fit(  self, server_round: int, results: List):
        '''The server aggregates the parameters (individuals) from the clients'''
        self.params_aggregation = results 
        self.server_round = server_round
        
    def aggregate_evaluation( self, results: List):
        '''The server finds the individual that in average has higher MI from the clients'''
        individuals=self.params_aggregation
        average_result=[]
        for individual in individuals:
            average_result.append(np.mean([result[str(individual)] for result in results]))  
        print(max(average_result))   
        best_individual_index=average_result.index(max(average_result))
        best_individual=[elem for elem in individuals[best_individual_index]]
        self.global_bayesian_network = convert_individual_to_network(best_individual,self.attributes_metadata['all_attributes'])
        return best_individual
    
    def aggregate_conditional_probabilities(self, conditional_probabilities_clients: List, keys:List):
        '''The server computes the average of the conditional probabilities from the clients'''
        result=[]
        for elements in zip(*conditional_probabilities_clients):
            result.append(np.sum(elements,axis=0)/self.number_of_clients)
        conditional_probabilities_dict={}
        all_keys = list(nested_dict_keys(self.describer.data_description['conditional_probabilities']))
        root = self.global_bayesian_network[0][1][0]
        nested_set(conditional_probabilities_dict, root.split('.'), result[0].tolist())
        for idx, (child, parents) in enumerate(self.global_bayesian_network):
            keys_child = [element for element in all_keys if child in element]
            values=result[idx+1].tolist()
            for i,key in enumerate(keys_child):
                nested_set(conditional_probabilities_dict, key.split('.'), values[i])
        self.conditional_probabilities = conditional_probabilities_dict
 
        
    def aggregate_frequencies_and_get_conditional_probabilities(self, frequencies_clients: List):
        ''' The server aggregates the (noisy) frequencies from the clients and computes the conditional probabilities'''
        
        k = len(self.global_bayesian_network[-1][1])
        conditional_distributions = {}
        
        # first k+1 attributes
        root = self.global_bayesian_network[0][1][0]
        kplus1_attributes = [root]
        for child, _ in self.global_bayesian_network[:k]:
            kplus1_attributes.append(child)

        params_noisy_kplus1_attributes = [ elem[0] for elem in frequencies_clients] 
        check_all_arrays_have_same_size(params_noisy_kplus1_attributes)
             
        stats=pd.DataFrame(frequencies_clients[0][0][:,np.arange(k+1)],columns=kplus1_attributes)
        
        aggregated_frecuencies = aggregate_counts(params_noisy_kplus1_attributes,k) 
        stats['count'] = aggregated_frecuencies
        noisy_dist_of_kplus1_attributes = stats

        root_stats = noisy_dist_of_kplus1_attributes.loc[:, [root, 'count']].groupby(root).sum()['count']
        conditional_distributions[root] = normalize_given_distribution(root_stats).tolist()

        for idx, (child, parents) in enumerate(self.global_bayesian_network):
            
            conditional_distributions[child] = {}
            
            if idx < k:
                stats = noisy_dist_of_kplus1_attributes.copy().loc[:, parents + [child, 'count']]
                stats = stats.groupby(parents + [child], as_index=False,observed=False).sum()        
            else:
                params_noisy_attributes = [ elem[idx-k+1] for elem in frequencies_clients] 
                check_all_arrays_have_same_size(params_noisy_attributes)
                stats=pd.DataFrame(frequencies_clients[0][idx-k+1][:,np.arange(k+1)],columns=parents + [child])
                aggregated_frecuencies = aggregate_counts(params_noisy_attributes,k) 
                stats['count'] = aggregated_frecuencies
                stats = stats.loc[:, parents + [child, 'count']]
            
            for parents_instance, stats_sub in stats.groupby(parents):
                stats_sub = stats_sub.sort_values(by=child)
                dist = normalize_given_distribution(stats_sub['count']).tolist()

                if len(parents) == 1:
                    parents_key = str([parents_instance])
                else:
                    parents_key = str(list(parents_instance))

                conditional_distributions[child][parents_key] = dist
            
        self.conditional_probabilities=conditional_distributions
                        

def fix_array_to_sum_to_1(arr):
    # Convert each element in the array to Decimal objects
    decimal_array = [Decimal(str(val)) for val in arr]
    
    # Set the desired precision (number of decimal places)
    getcontext().prec = 28
    
    # Calculate the sum of the decimal array
    sum_decimal_array = sum(decimal_array)
    
    # Normalize the decimal array to ensure it sums to exactly 1
    normalized_array = [val / sum_decimal_array for val in decimal_array]
    
    normalized_array = [ float(val) for val in decimal_array]
    
    return normalized_array        
    
def check_all_arrays_have_same_size(array_list):
    # Check if all arrays in the list have the same shape
    assert all(np.array_equal(arr.shape, array_list[0].shape) for arr in array_list[1:]), "Arrays in the list must have the same shape"