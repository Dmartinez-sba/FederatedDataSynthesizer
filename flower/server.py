import argparse
import sys
import os
import pickle
from typing import Callable, Dict, List, Optional, Tuple, Union

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
import json

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Metrics,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.common.typing import GetParametersIns, GetPropertiesIns

from DataSynthesizer.federated_utils.server import fix_array_to_sum_to_1
from DataSynthesizer.lib.utils import aggregate_counts, convert_individual_to_network, nested_dict_keys,deep_get_from_dict,nested_set, normalize_given_distribution, verify_all_arrays_have_same_size

class FedGA(FedAvg):
    def __init__(
        self,     
        fraction_fit: float = 0.1,
        fraction_evaluate: float = 0.1,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        attributes_metadata: Dict = None, 
        encoding: Dict = None,
        global_conditional_probabilities: Dict = None,
        genetic_algorithm_parameters: Dict = None,
        total_number_of_rounds: int = 1,
        aggregation_type: str = 'frequencies',
        threshold: int = None,
        seed: int = 0
    ) -> None:
        """Federated Genetic Algorithm for PrivBayes

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        attributes_metadata: Dict
            Dictionary describing the column names and types in the clients. Defaults to None. 
        encoding: Dict
            Dictionary containing the parameters for encoding numeric and categorical columns. Defaults to None. 
        global_conditional_probabilities: Dict 
            Dictionary containing the global conditional probabilities of the global BN. Defaults to None. 
        genetic_algorithm_parameters: Dict 
            Dictionary containing the parameters used in the genetic algorithm. Defaults to None. 
        total_number_of_rounds : int
            Total number of rounds of the federated learning training. Defaults to 1.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters  
        )
        self.attributes_metadata = attributes_metadata
        self.common_encoding = encoding
        self.global_conditional_probabilities = global_conditional_probabilities
        self.genetic_algorithm_parameters = genetic_algorithm_parameters
        self.total_number_of_rounds=total_number_of_rounds
        self.aggregation_type = aggregation_type
        self.threshold = threshold
        self.train_seed = seed

    def initialize_parameters(
        self, client_manager: Optional[ClientManager] = None
    ) -> Optional[Parameters]:
        print('Initializing parameters...')
        # Ask a random client for all column names and data types.
        random_client = client_manager.sample(1)[0]
        ins = GetPropertiesIns(config={'metadata': True})
        get_properties_res = random_client.get_properties(ins=ins, timeout=30.0)
        metadata_dict = json.loads(get_properties_res.properties['metadata'])
        self.attributes_metadata = metadata_dict
        encoding = {}
        for attr in self.attributes_metadata['all_attributes']:
             if attr in self.attributes_metadata['numerical_attributes']:
                encoding[f'{attr}']= ['min','max','missing_rate']
             else:
                encoding[f'{attr}']=['distribution_bins','missing_rate']
        # Ask all clients for the minimum and maximum of numerical attributes and categories in the categorical columns.
        ins = GetPropertiesIns(config={'encoding': json.dumps(encoding)})
        clients = client_manager.sample(num_clients=self.min_available_clients, min_num_clients=self.min_available_clients)
        clients_general_info=[]
        for client in clients:
            get_encoding_res = client.get_properties(ins=ins, timeout=90.0)
            dict_encoding_client = json.loads(get_encoding_res.properties['encoding'])
            clients_general_info.append(dict_encoding_client)
        # Calculate the global minimum and maximum of numerical attributes and common categories for categorical attributes
        common_encoding=configure_common_encoding(clients_general_info)
        self.global_missing_rate = configure_global_missing_rate(clients_general_info)
        self.common_encoding = common_encoding
        print(common_encoding)
        return ndarrays_to_parameters([]) 

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        print(f'the server is calling config fit in the round:{server_round}')
                
        # The config Dict contains the parameters shared with local clients to perform the GA algorithm locally. 
        config = {
                    'learning_phase': 'structure_learning' if server_round <= self.total_number_of_rounds-1 else 'parameter_learning',
                    'server_round': server_round, 
                    'local_epochs': self.genetic_algorithm_parameters['aggregation_interval'] if server_round < self.total_number_of_rounds-1 else 0, 
                    'epsilon': self.genetic_algorithm_parameters['epsilon'],
                    'k': self.genetic_algorithm_parameters['degree_of_bayesian_network'],
                    'source_genes': self.genetic_algorithm_parameters['source_genes'],
                    'genepool_size': self.genetic_algorithm_parameters['genepool_size'], 
                    'size': self.min_available_clients,
                    'evaluate_individuals': True if server_round == self.total_number_of_rounds-1 else False,
                    'aggregation_type': self.aggregation_type,
                    'threshold': self.threshold,
                    'training_seed': self.train_seed

                }
        common_encoding = json.dumps(self.common_encoding)
        global_missing_rate = json.dumps(self.global_missing_rate)
        config['encoding']= common_encoding 
        config['missing_rate']= global_missing_rate 
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(num_clients=self.min_available_clients, min_num_clients=self.min_available_clients)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(f'The server is calling aggregate fit in the round:{server_round}')
        
        if not results:
            return None, {}
        
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        # Get results
        values_aggregated = [
            (parameters_to_ndarrays(fit_res.parameters)) for _, fit_res in results
        ]

         # Get metrics
        metrics_aggregated = [
            (fit_res.metrics) for _, fit_res in results
        ]

        print(server_round)
        
        if server_round <= self.total_number_of_rounds-1:
            # Structure Learning Phase
            individuals_aggregated=[]
            for individual in values_aggregated:
                individuals_aggregated.append(individual[0])
                for elem in individual[1:]: 
                    individuals_aggregated.append(elem)

            if len(metrics_aggregated[0])>0:
                best_individual_in_avg = aggregate_evaluation(metrics_aggregated,values_aggregated)
                self.global_bayesian_network = convert_individual_to_network(best_individual_in_avg,self.attributes_metadata['all_attributes'])
                print(self.global_bayesian_network)
                return ndarrays_to_parameters(best_individual_in_avg), {}           
        else:
            # Parameter Learning Phase
            print('Parameter Learning Phase')
            if 'frequencies' in self.aggregation_type:
                self.global_conditional_probabilities = aggregate_frequencies_and_get_conditional_probabilities(self.global_bayesian_network, values_aggregated) 
                return [],{}
            else:
                if len(metrics_aggregated[0])>0:
                    if metrics_aggregated[0].get('conditional_probabilities') != None:
                        list_conditional_probabilities = [metrics.get('conditional_probabilities') for metrics in metrics_aggregated]
                        self.global_conditional_probabilities=aggregate_conditional_probabilities(list_conditional_probabilities)
                        return [], {}         
                        
                            
         #if len(metrics_aggregated[0])>0:
         #   if metrics_aggregated[0].get('conditional_probabilities') != None:
         #       list_conditional_probabilities = [metrics.get('conditional_probabilities') for metrics in metrics_aggregated]
         #       self.global_conditional_probabilities=aggregate_conditional_probabilities(list_conditional_probabilities)
         #       return [], {}
         #   else: 
         #       # Structure Learning Phase (Evaluation- Best Individual)  
         #       best_individual_in_avg = aggregate_evaluation(metrics_aggregated,values_aggregated)
         #      self.global_bayesian_network = convert_individual_to_network(best_individual_in_avg,self.attributes_metadata['all_attributes'])
         #        print(self.global_bayesian_network)
         #        return ndarrays_to_parameters(best_individual_in_avg), {}   
        return ndarrays_to_parameters(individuals_aggregated), {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Let's assume we won't perform the global model evaluation on the server side.
        return None
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        
        if server_round == self.total_number_of_rounds: 

            config = {'conditional_probabilities': json.dumps(self.global_conditional_probabilities)}
            evaluate_ins = EvaluateIns(parameters, config)
            
            clients = [client_manager.sample(1)[0]]
            
            return [(client, evaluate_ins) for client in clients]
             
    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""
        return None, {}
    
def deserialize_dict_list_values(dictionary: Dict):
    new_dict={}
    for property, val in  dictionary.items():
        new_dict[property] = fl.common.bytes_to_ndarray(val)[0]  
    return new_dict 

def serialize_dict_list_values(dictionary: Dict):
    new_dict={}
    for property, val in  dictionary.items():
        new_dict[property] = fl.common.ndarray_to_bytes([val])  
    return new_dict 

def configure_common_encoding(clients_info: List[Dict]):
    common_dict={}
    for attr, attr_dict in clients_info[0].items():
        if 'min' in attr_dict:
            list_min=list(map(lambda d: d[attr]['min'],  clients_info))
            list_max=list(map(lambda d: d[attr]['max'],  clients_info))
            min_attr= min(list_min)
            max_attr= max(list_max)
            common_dict[attr]=[min_attr,max_attr]
        else:
            categories_per_attribute = list(map(lambda d: set(d[attr]['distribution_bins']),  clients_info))
            all_categories_per_attribute=set().union(*categories_per_attribute)
            categories= list(all_categories_per_attribute)
            categories.sort()
            common_dict[attr]= categories
    return common_dict

def configure_global_missing_rate(clients_info: List[Dict]):
    common_dict={}
    for attr, attr_dict in clients_info[0].items():
        missing_rates= list(map(lambda d: d[attr]['missing_rate'],  clients_info))
        total_missing_rate = sum(missing_rates)
        common_dict[attr]= total_missing_rate
    return common_dict


#def aggregate_conditional_probabilities(self, conditional_probabilities_clients: List, all_keys:List):
#        '''The server computes the average of the conditional probabilities from the clients'''
#        result=[]
#        for elements in zip(*conditional_probabilities_clients):
#            result.append(np.sum(elements,axis=0)/self.number_of_clients)
#        conditional_probabilities_dict={}
#        root = self.global_bayesian_network[0][1][0]
#       nested_set(conditional_probabilities_dict, root.split('.'), result[0].tolist())
#        for idx, (child, parents) in enumerate(self.global_bayesian_network):
#            keys_child = [element for element in all_keys if child in element]
#            values=result[idx+1].tolist()
#            for i,key in enumerate(keys_child):
#                nested_set(conditional_probabilities_dict, key.split('.'), values[i])
#        self.conditional_probabilities = conditional_probabilities_dict

def aggregate_conditional_probabilities(conditional_probabilities_clients: List):
    '''Function to aggregate the conditional probabilities from all the clients'''
    transform_conditional_probabilities=[json.loads(conditional_prob) for conditional_prob in conditional_probabilities_clients]
    global_conditional_probabilities={}
    all_nested_keys = list(nested_dict_keys(transform_conditional_probabilities[0]))
    for nested_key in all_nested_keys:
        output_list=[deep_get_from_dict(i,nested_key) for i in transform_conditional_probabilities]
        output_list=list(filter(lambda item: item is not None, output_list))
        result=[sum(x) for x in zip(*output_list)]
        result=[elem/len(conditional_probabilities_clients) for elem in result]
        result=fix_array_to_sum_to_1(result)
        nested_set(global_conditional_probabilities,nested_key.split('.'),result)
    return global_conditional_probabilities

def aggregate_frequencies_and_get_conditional_probabilities(bayesian_network, frequencies_clients: List):
        ''' Function to aggregate (noisy) frequencies from the clients and compute the conditional probabilities'''
        
        k = len(bayesian_network[-1][1])
        conditional_distributions = {}
        
        # first k+1 attributes
        root = bayesian_network[0][1][0]
        kplus1_attributes = [root]
        for child, _ in bayesian_network[:k]:
            kplus1_attributes.append(child)

        params_noisy_kplus1_attributes = [ elem[0] for elem in frequencies_clients] 
        verify_all_arrays_have_same_size(params_noisy_kplus1_attributes)
             
        stats=pd.DataFrame(frequencies_clients[0][0][:,np.arange(k+1)],columns=kplus1_attributes)
        
        aggregated_frecuencies = aggregate_counts(params_noisy_kplus1_attributes,k) 
        stats['count'] = aggregated_frecuencies
        noisy_dist_of_kplus1_attributes = stats

        root_stats = noisy_dist_of_kplus1_attributes.loc[:, [root, 'count']].groupby(root).sum()['count']
        conditional_distributions[root] = normalize_given_distribution(root_stats).tolist()

        for idx, (child, parents) in enumerate(bayesian_network):
            
            conditional_distributions[child] = {}
            
            if idx < k:
                stats = noisy_dist_of_kplus1_attributes.copy().loc[:, parents + [child, 'count']]
                stats = stats.groupby(parents + [child], as_index=False,observed=False).sum()        
            else:
                params_noisy_attributes = [ elem[idx-k+1] for elem in frequencies_clients] 
                print(f'parents and child are: {parents},{child}')
                shape_general=params_noisy_attributes[0].shape
                verify_all_arrays_have_same_size(params_noisy_attributes)
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
            
        return conditional_distributions
    
        
def aggregate_evaluation( results: List, individuals: List):
    '''Function to select best individual given dictionary of scores'''
    average_result=[]
    for individual in individuals:
        transformed_individual= [elem.tolist() for elem in individual]
        average_result.append(np.mean([result[str(transformed_individual)] for result in results]))     
    best_individual_index=average_result.index(max(average_result))
    best_individual=[elem for elem in individuals[best_individual_index]]
    return best_individual

def main():
    
    parser = argparse.ArgumentParser(description="Flower Server")
    
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=20,
        required=False,
        help="Specifies the number of federated rounds",
    )

    args = parser.parse_args()
     
    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=FedGA(min_available_clients=3,min_evaluate_clients=1),
    )

if __name__ == "__main__":
    main()