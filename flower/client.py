import sys
import os
import pickle
from DataSynthesizer.lib.utils import set_classes_from_attributes_to_df, dict_to_list


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import warnings
from typing import Dict, List, Tuple
import json

import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import flwr as fl

from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.lib.PrivBayes import construct_noisy_conditional_distributions, get_all_noisy_frequencies, get_modify_frequencies_with_threshold
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.federated_utils.partitions import partition_dataset
from DataSynthesizer.lib.utils import nested_dict_keys

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, describer, traindata, categorical_attributes,client_input_dir, client_output_dir, local_save=False):
        self.cid = cid
        self.describer = describer
        self.traindata = traindata
        self.categorical_attributes = categorical_attributes
        self.client_input_dir = client_input_dir
        self.client_output_dir = client_output_dir
        self.local_save = local_save
        self.params = [] # Depends on the learning phase. 
        

    def get_parameters(self):
        print(f"Client get_parameters")
        return self.params

    def get_properties(self, config):
       print(f'Getting properties from client {self.cid}')
       dict_properties ={}
       for prop in config.keys():
          if 'metadata' in prop:
            dict_properties['metadata'] = json.dumps(self.get_minimum_attributes_metadata())
          elif 'encoding' in prop:
            encoding_dict = {}
            encoding = json.loads(config['encoding'])
            for attr,properties in encoding.items():
                attr_dict=self.get_info_attributes(attr,properties)
                encoding_dict.update(attr_dict)
            dict_properties['encoding'] = json.dumps(encoding_dict)
          else:
            prop_dict={}
            dict_properties.update(prop_dict)
       return dict_properties
   
    def update_final_model(self,individual,config):
        str_df = self.describer.df_encoded
        str_df = set_classes_from_attributes_to_df(str_df,self.describer.data_description['attribute_description'])
        bn=self.describer.ga_network_builder.convert_to_network(individual, str_df)[1:]
        self.describer.data_description['bayesian_network'] = bn
        print(f'Client {self.cid} updating final model')
        if config['aggregation_type'] == 'frequencies':
            frequencies = get_all_noisy_frequencies(bn, str_df, config['epsilon'],partial_noise=True,number_of_clients=config['size'])
            self.params = frequencies
        elif config['aggregation_type'] == 'sfrequencies': 
            frequencies = get_modify_frequencies_with_threshold(bn, str_df, config['threshold'],method='suppression')
            self.params = frequencies
        elif config['aggregation_type'] == 'mfrequencies': 
            frequencies = get_modify_frequencies_with_threshold(bn, str_df, config['threshold'],method='perturbation')
            self.params = frequencies
        elif config['aggregation_type'] == 'rfrequencies': 
            frequencies = get_modify_frequencies_with_threshold(bn, str_df, config['threshold'],method='rounding')
            self.params = frequencies
        else:
            # Only this mode shares the (locally noised) CPDs; the frequency modes share counts only.
            self.describer.data_description['conditional_probabilities'] = construct_noisy_conditional_distributions( bn, str_df, config['epsilon'])
            list_conditional_probabilities = dict_to_list(self.describer.data_description['conditional_probabilities'])
            list_conditional_probabilities= [np.array(element) if not isinstance(element, np.ndarray) else element for element in list_conditional_probabilities]
            self.params = list_conditional_probabilities
        
   
    def set_parameters(self, parameters, config):
        if len(parameters)>1:
            transformed_parameters=[]
            for val in parameters:
                transformed_parameters.append(val.tolist())
            split_size=len(transformed_parameters)//config['size']
            individuals = [transformed_parameters[i:i + split_size] for i in range(0, len(transformed_parameters), split_size)]
            self.params = individuals

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        print(f"calling client fit in round {config['server_round']}") 
        metrics={}
        if config['server_round']==1:
            self.set_global_attributes_info(json.loads(config['encoding']))
            self.set_global_missing_rate(json.loads(config['missing_rate']))
            self.set_distributions()
        if config['learning_phase'] == 'structure_learning': 
            self.set_parameters(parameters,config)
            self.describer.describe_dataset_in_correlated_attribute_mode_ga_locally(  
                                                            federated_round=config['server_round'],
                                                            individuals= self.params,
                                                            epsilon=config['epsilon'],
                                                            k=config['k'],
                                                            source_genes=config['source_genes'],
                                                            genepool_size=config['genepool_size'],
                                                            epochs=config['local_epochs'],
                                                            seed=self.cid,
                                                            )
            best_individual=self.describer.ga_network_builder.best_individual
            parameters=best_individual
        if config['evaluate_individuals']:
            metrics=self.evaluate_individuals(self.params)
        if config['learning_phase'] == 'parameter_learning':
            self.update_final_model(parameters,config)
            self.get_parameters()
            parameters=self.params
            if 'frequencies' not in config['aggregation_type']:
                metrics = {'conditional_probabilities':json.dumps(self.describer.data_description['conditional_probabilities'])}
        if self.local_save:
            self.save()        
        return (parameters, 0, metrics)
    
    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        """Evaluate parameters on the locally held test set."""
        self.describer.data_description['conditional_probabilities'] = json.loads(config['conditional_probabilities'])
        description_file_ga = os.path.join(self.client_output_dir, "ga_corrmeta_fed.json")
        self.describer.save_dataset_description_to_file(description_file_ga)
        return 0.0, 0, {}
    
    
    def get_minimum_attributes_metadata(self):
        ''' Function to obtain the list of columns in the local dataset and the type (numerical or categorical)'''
        metadata = {}
        all_attributes = self.describer.data_description['meta']['all_attributes']
        numerical_attributes = [ col for col in self.traindata.columns if col not in self.categorical_attributes] 
        categorical_attributes = self.categorical_attributes 
        metadata['all_attributes'] = all_attributes
        metadata['numerical_attributes'] = numerical_attributes
        metadata['categorical_attributes']= categorical_attributes
        return metadata
    
    def get_info_attributes(self, attr, properties):
        ''' Function to obtain given properties of a specific column''' 
        attribute_description_dict = self.describer.data_description['attribute_description']
        dict_info = {}
        for property in properties:
            dict_info.update({ property: attribute_description_dict[attr][property]})      
        attribute_info = {attr: dict_info}
        return attribute_info 
    
    def set_global_attributes_info(self,global_config):
        ''' Function to update the min-max of numerical attributes and the values of categorical variables'''          
        for column in self.describer.attr_to_column.values():
            attr_name = column.name
            if attr_name in self.categorical_attributes:
                column.is_categorical = True
                column.distribution_bins = np.array(global_config[f'{attr_name}'])
            else:
                column.is_categorical = False
                column.min = global_config[f'{attr_name}'][0]
                column.max = global_config[f'{attr_name}'][1]
                column.distribution_bins = np.array([global_config[f'{attr_name}'][0],global_config[f'{attr_name}'][1]])      
        # record attribute information in json format
        self.describer.data_description['attribute_description'] = {}
        for attr, column in self.describer.attr_to_column.items():
            self.describer.data_description['attribute_description'][attr] = column.to_json()
            
    def set_global_missing_rate(self,global_config):
        for column in self.describer.attr_to_column.values():
            attr_name = column.name
            column.missing_rate = global_config[f'{attr_name}']
        # record attribute information in json format
        self.describer.data_description['attribute_description'] = {}
        for attr, column in self.describer.attr_to_column.items():
            self.describer.data_description['attribute_description'][attr] = column.to_json()

    def set_distributions(self):
        '''Function to calculate the distribution bins probability locally'''
        self.describer.describe_dataset_in_independent_attribute_mode_locally(epsilon=0)

    def evaluate_individuals(self, individuals):
        '''Function to evaluate a group of individuals representing BNs locally'''
        str_df = self.describer.df_encoded
        evaluation_tuples = self.describer.ga_network_builder.eval_genepool(individuals,str_df)
        dict_evaluated_individuals={}
        for i,individual_tuple in enumerate(evaluation_tuples):
            dict_evaluated_individuals[str(individual_tuple[1])]=individual_tuple[0]
        return dict_evaluated_individuals

    def save(self):
        '''Function to save the describer object from a client'''
        filepath=os.path.join(self.client_input_dir,f'describer_client{self.cid}')
        with open(filepath,'wb') as outfile:
            pickle.dump(self.describer, outfile)


def get_client_fn(clients_partitions: Dict, train_data: pd.DataFrame, save_dir: str, **kwargs):
    """Return a function that creates a Flower client."""
    
    def client_fn(cid) -> FlowerClient:
        elem = clients_partitions[int(cid)]
        local_data = train_data.filter(items = elem, axis=0)
        #data_file=f'{save_dir}/data_client{cid}.csv'
        describer_file=f'{save_dir}/describer_client{cid}'
        if os.path.exists(describer_file):
            with open(describer_file, 'rb') as f:
                describer = pickle.load(f)
        else:
            describer = DataDescriber(category_threshold=kwargs['category_threshold'])
            describer.describe_dataset_in_random_mode(df_input=local_data,
                                                    attribute_to_is_categorical=kwargs['categorical_attributes'],
                                                    attribute_to_is_candidate_key=kwargs['candidate_keys']
                                                    )
        client = FlowerClient(cid, describer, train_data, kwargs['list_categorical_attributes'] , save_dir, save_dir, True) 
        return client 
    
    return client_fn

