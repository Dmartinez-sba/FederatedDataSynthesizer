from typing import Dict, List
import pandas as pd  
import numpy as np
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.lib.PrivBayes import construct_noisy_conditional_distributions, get_all_noisy_frequencies, get_modify_frequencies_with_threshold
from DataSynthesizer.lib.utils import  dict_to_list, set_classes_from_attributes_to_df

class FedClient:
    def __init__(self, id, data, describer, epsilon = 0, k=2, source_genes=10, genepool_size=100):
        self.id = id
        self.data: pd.DataFrame = data
        self.data_encoded = None
        self.current_population = None
        self.describer: DataDescriber = describer
        self.epsilon = epsilon
        self.degree_of_bayesian_network = k
        self.source_genes = source_genes
        self.genepool_size = genepool_size
        self.frequencies = []
        self.modified_frecuencies = []
        self.conditional_probabilities_params = []
        
    def __str__(self):
        return f'Federated Client with id:{self.id}'
        
    def get_min_max_numerical_attributes(self, attributes: List):
        ''' Function to obtain the min-max for each numerical column''' 
        attribute_description_dict = self.describer.data_description['attribute_description']
        numerical_attributes_info = { attr: {'min': attribute_description_dict[attr]['min'], 'max': attribute_description_dict[attr]['max']}  for attr in attributes}
        return numerical_attributes_info
    
    def get_values_from_categorical_attributes(self, attributes: List): 
        ''' Function to obtain the categories for each categorical column'''
        attribute_description_dict = self.describer.data_description['attribute_description']
        categorical_attributes_info = { attr: {'categories': set(attribute_description_dict[attr]['distribution_bins'])}  for attr in attributes}
        return categorical_attributes_info
    
    def get_minimum_attributes_metadata(self):
        ''' Function to obtain the list of columns in the local dataset and the type (numerical or categorical)'''
        metadata = {}
        all_attributes = self.describer.data_description['meta']['all_attributes']
        attribute_description_dict = self.describer.data_description['attribute_description']
        numerical_attributes = list(filter(lambda x: 'is_categorical' in attribute_description_dict[x] and not attribute_description_dict[x]['is_categorical'], attribute_description_dict))
        categorical_attributes = list(filter(lambda x: 'is_categorical' in attribute_description_dict[x] and attribute_description_dict[x]['is_categorical'], attribute_description_dict))
        metadata['all_attributes'] = all_attributes 
        metadata['numerical_attributes'] = numerical_attributes
        metadata['categorical_attributes']= categorical_attributes
        return metadata
    
 
    def set_global_attributes_info(self,global_info_attributes: Dict):
        ''' Function to update the min-max of numerical attributes and the values of categorical variables'''          
        for column in self.describer.attr_to_column.values():
            attr_name = column.name
            if attr_name in global_info_attributes['categorical_attributes']:
                column.distribution_bins = np.array(global_info_attributes['attributes_description'][attr_name]['distribution_bins'])
            elif attr_name in global_info_attributes['numerical_attributes']:
                column.min = global_info_attributes['attributes_description'][attr_name]['min']
                column.max = global_info_attributes['attributes_description'][attr_name]['max']
                column.distribution_bins = np.array(global_info_attributes['attributes_description'][attr_name]['distribution_bins'])
            else:
                column.infer_domain()        
        # record attribute information in json format
        self.describer.data_description['attribute_description'] = {}
        for attr, column in self.describer.attr_to_column.items():
            self.describer.data_description['attribute_description'][attr] = column.to_json()
            
    def set_distributions(self):
        ''' Function to describe in independent mode locally'''
        self.describer.describe_dataset_in_independent_attribute_mode_locally(epsilon=self.epsilon)
        
    def set_conditional_probabilities_as_params(self):
        ''' Function to convert the dict of consitional probabilities to a list of numpy ndarrays'''
        list_conditional_probabilities = dict_to_list(self.describer.data_description['conditional_probabilities'])
        list_conditional_probabilities= [np.array(element) if not isinstance(element, np.ndarray) else element for element in list_conditional_probabilities]
        self.conditional_probabilities_params = list_conditional_probabilities
        
    def fit(self, epochs, federated_round, individuals, server_params={}):
        ''' Function to train the genetic algorithm locally '''
        individuals = [ ind for ind in individuals if ind != self.describer.ga_network_builder.best_individual]
        self.describer.describe_dataset_in_correlated_attribute_mode_ga_locally(  
                                                    federated_round=federated_round,
                                                    individuals= individuals,
                                                    epsilon=self.epsilon,
                                                    k=self.degree_of_bayesian_network,
                                                    source_genes=self.source_genes,
                                                    genepool_size=self.genepool_size,
                                                    epochs=epochs,
                                                    seed=self.id,
                                                    )
        best_individual=self.describer.ga_network_builder.best_individual
        return best_individual
        
    def evaluation(self, individuals):
        ''' Function to evaluate the individuals received from the server'''
        str_df = self.describer.df_encoded
        evaluation_tuples = self.describer.ga_network_builder.eval_genepool(individuals,str_df)
        dict_evaluated_individuals={}
        for i,individual_tuple in enumerate(evaluation_tuples):
            dict_evaluated_individuals[str(individual_tuple[1])]=individual_tuple[0]
        return dict_evaluated_individuals

    
    def update_final_model(self,individual, number_of_clients):
        ''' Function to update the final model '''
        str_df = self.describer.df_encoded
        bn=self.describer.ga_network_builder.convert_to_network(individual, str_df)[1:]
        str_df = set_classes_from_attributes_to_df(str_df,self.describer.data_description['attribute_description'])
        self.describer.data_description['bayesian_network'] = bn 
        self.describer.data_description['conditional_probabilities'] = construct_noisy_conditional_distributions( bn, str_df, self.epsilon / 2)
        self.set_conditional_probabilities_as_params()
        frequencies = get_all_noisy_frequencies(bn, str_df, self.epsilon,partial_noise=True, number_of_clients=number_of_clients)
        modified_frequencies = get_modify_frequencies_with_threshold(bn, str_df, self.epsilon)
        self.frequencies = frequencies
        self.modified_frecuencies = modified_frequencies
    
    def update_parameters(self,conditional_probabilities: Dict):
        ''' Function to update the conditional probabilities aggregated in the server'''
        self.describer.data_description['conditional_probabilities'] = conditional_probabilities
        # conditional_probabilities_dict={}
        # all_keys = list(nested_dict_keys(self.describer.data_description['conditional_probabilities']))
        # root = self.describer.data_description['bayesian_network'][0][1][0]
        # nested_set(conditional_probabilities_dict, root.split('.'), conditional_probabilities[0].tolist())
        # for idx, (child, parents) in enumerate(self.describer.data_description['bayesian_network']):
        #     keys_child = [element for element in all_keys if child in element]
        #     values=conditional_probabilities[idx+1].tolist()
        #     for i,key in enumerate(keys_child):
        #         nested_set(conditional_probabilities_dict, key.split('.'), values[i])
        # self.describer.data_description['conditional_probabilities'] = conditional_probabilities_dict


def client_fn(id: int, file: str, threshold_value: int = 20, categorical_attributes: Dict ={}, candidate_keys: Dict ={}, epsilon= 0, k=2, source_genes=10, genepool_size=200) -> FedClient:
    """Create a Flower client representing a single organization."""
    # Load model architecture (If necessary)

    # Load data 
    dataloader = pd.read_csv(file)
    
    # Describe the data in each of the clients (Datatypes)
    describer = DataDescriber(category_threshold=threshold_value)
    describer.describe_dataset_in_random_mode(dataset_file=file,
                                              attribute_to_is_categorical=categorical_attributes,
                                              attribute_to_is_candidate_key=candidate_keys
                                              )

    # Create a  single Flower client representing a single organization
    return FedClient(id, dataloader, describer, epsilon=epsilon, k=k, source_genes= source_genes, genepool_size= genepool_size)
        