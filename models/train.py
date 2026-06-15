import os 
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import wandb

sys.path.append('.') # to import local DataSynthesizer 

from omegaconf import OmegaConf

import argparse
import synthgauge as sg
import tempfile
import flwr as fl
import pandas as pd
import ray

from evaluation.utils import encode_categorical_columns
from evaluation.metrics.bivariate_fidelity import pairwise_correlation_difference
from evaluation.metrics.univariate_fidelity import average_hellinger_distance, sdv_metrics
from evaluation.metrics.privacy_distance import distance_to_closest_record
from evaluation.evaluation import evaluate_multiple_dimension_synthetic_data

from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator

from DataSynthesizer.federated_utils.partitions import partition_dataset_pandas, plot_partition_plotly, repeated_holdout
from evaluation.metrics.multivariate_fidelity import log_cluster_metric

from flower.client import get_client_fn
from flower.server import FedGA

def load_yaml_to_dict(file_path):
    """Reads yaml file and converts it to dictionary.

    Parameters:
    -----------
        file_path (str): 
            Path to yaml file with configuration parameters.

    Returns:
        config_dict (Dict): 
            Dictionary with configuration parameters.
    """
    config = OmegaConf.load(file_path)
    config_dict = OmegaConf.to_container(config)
    return config_dict

def generate_synthetic_data(model_path,size,seed=0):
    '''
    Generates synthetic data based on the Bayesian network description.
    
    Parameters
    ----------
    model_path (str):
      Path to the description file of the BN model.
    size(int): 
      Number of samples to generate.
    seed (int): 
      Random seed. 
    
    Returns
    ---------
    synthetic_data (pandas.DataFrame):
      DataFrame with synthetic data generated.
        
    '''
    start= time.time()
    ga_generator = DataGenerator()
    synthetic_data=ga_generator.generate_dataset_in_correlated_attribute_mode(size, model_path, seed)
    end=time.time()
    print(f'Time for synthetic data generation: {end-start}')
    return synthetic_data

def centralized_training(config, train_data, size=None, seed=0, save_dir='./',description_file_name='ga_corrmeta_cen.json'):
    """ Runs the centralized Bayesian Network with a genetic algorithm.

    Parameters:
    -----------
        config (Dict): 
            Configuration dict with parameters for the training.
        train_data (pandas.DataFrame): 
            Dataframe with the training data. 
        seed (int, optional): 
            Random seed. Defaults to 0.

    Returns:
    -------
        synthetic_data (pandas.DataFrame): 
            Dataframe with the synthetic data generated. 
    """  
    temp_dir=save_dir
    
    categorical_attributes = config.data_synthesizer.categorical_attributes

    cat_attributes = dict()
    for val in categorical_attributes:
        cat_attributes[val]= True
        

    description_file_cen_ga = os.path.join(temp_dir, description_file_name)
    
    
    if not Path(f'{temp_dir}/{description_file_name}').is_file():
        describer_ga = DataDescriber(category_threshold=config.data_synthesizer['threshold_value'],histogram_bins=config.data_synthesizer['histogram_bins'])
        describer_ga.describe_dataset_in_correlated_attribute_mode_ga(df_input=train_data,
                                                                    epsilon=config.genetic_algorithm['epsilon'],
                                                                    k=config.genetic_algorithm['degree_of_bayesian_network'],
                                                                    attribute_to_is_categorical=cat_attributes, # TO DO
                                                                    attribute_to_is_candidate_key={}, # TO DO
                                                                    source_genes=config.genetic_algorithm['source_genes'],
                                                                    genepool_size=config.genetic_algorithm['genepool_size'],
                                                                    epochs=config.epochs,
                                                                    seed=seed,
                                                                    )
        
        describer_ga.save_dataset_description_to_file(description_file_cen_ga)
        print("Wrote network to " + description_file_cen_ga) 
        
    if size == None:
        size = train_data.shape[0]
    
    synthetic_data_cen= generate_synthetic_data(description_file_cen_ga,size)
               
    return synthetic_data_cen

def centralized_training_original_priv_bayes(config, train_data, size=None, seed=0, save_dir='./',description_file_name='ga_corrmeta_cen.json'):
    """ Runs the centralized Bayesian Network with a genetic algorithm.

    Parameters:
    -----------
        config (Dict): 
            Configuration dict with parameters for the training.
        train_data (pandas.DataFrame): 
            Dataframe with the training data. 
        seed (int, optional): 
            Random seed. Defaults to 0.

    Returns:
    -------
        synthetic_data (pandas.DataFrame): 
            Dataframe with the synthetic data generated. 
    """  
    temp_dir=save_dir
    
    categorical_attributes = config.data_synthesizer.categorical_attributes

    cat_attributes = dict()
    for val in categorical_attributes:
        cat_attributes[val]= True
        

    description_file_cen= os.path.join(temp_dir, description_file_name)
    
    
    if not Path(f'{temp_dir}/{description_file_name}').is_file():
        describer_ga = DataDescriber(category_threshold=config.data_synthesizer['threshold_value'],histogram_bins=config.data_synthesizer['histogram_bins'])
        describer_ga.describe_dataset_in_correlated_attribute_mode(df_input=train_data,
                                                                    k=config.genetic_algorithm['degree_of_bayesian_network'],
                                                                    epsilon=config.genetic_algorithm['epsilon'],
                                                                    attribute_to_is_categorical=cat_attributes, # TO DO
                                                                    attribute_to_is_candidate_key={}, # TO DO
                                                                    seed=seed,
                                                                    )
        
        describer_ga.save_dataset_description_to_file(description_file_cen)
        print("Wrote network to " + description_file_cen) 
        
    if size == None:
        size = train_data.shape[0]
    
    synthetic_data_cen= generate_synthetic_data(description_file_cen,size)
               
    return synthetic_data_cen              


def local_training(config, train_data, clients_partition,seed=0,save_dir='./'):
    """ Runs the Bayesian Network with a genetic algorithm locally for each client.

    Parameters:
    -----------
        config (Dict): 
            Configuration dict with parameters for the training.
        train_data (pandas.DataFrame): 
            Dataframe with the training data. 
        seed (int, optional): 
            Random seed. Defaults to 0.

    Returns:
    -------
        synthetic_datasets (List): 
            List with dataframes corresponding to synthetic data generated at each client. 
    """
    synthetic_datasets=[]
    df=train_data.copy(deep=True)
    size=train_data.shape[0]

    
    for elem in clients_partition.items():
        local_data = df.filter(items = elem[1], axis=0).copy(deep=True)
        synthetic_data = centralized_training(config,local_data,size=local_data.shape[0],seed=seed,save_dir=save_dir,description_file_name=f'ga_corrmeta_local_{elem[0]}.json')
        synthetic_datasets.append(synthetic_data)
               
    return synthetic_datasets
           

def federated_training(config, train_data, py_modules, number_of_clients, clients_partition, save_dir='../results', seed=0):
    """ Runs the federated Bayesian Network with a genetic algorithm.

    Parameters
    -----------
        config (Dict): 
            Configuration dict with parameters for the training.
        train_data (pandas.DataFrame): 
            Dataframe with the training data. 
        py_modules (list):
            List to the folders of libraries used by Flower Strategy.
        number_of_clients (int):
            Number of clients in the federated training.
        clients_partition (Dict):
            Dictionary to assign data to each client.

    Returns
    --------
        synthetic_data (pandas.DataFrame): 
            Dataframe with the synthetic data generated. 
    """

    # Must be before Ray starts
    #os.environ["RAY_memory_usage_threshold"] = "0.98"
        
    # Parse arguments for DataSynthesizer
    if 'categorical_attributes' not in config.data_synthesizer: 
        categorical_attributes = [col for col in train_data.columns if train_data[col].dtype=="O"]
    else:
        categorical_attributes = list(config.data_synthesizer['categorical_attributes'])

    if 'candidate_keys' not in config.data_synthesizer:
         candidate_keys = []
    else: 
         candidate_keys = list(config.data_synthesizer['candidate_keys'])
    
    cat_attributes = dict()
    for val in categorical_attributes:
        cat_attributes[val]= True

    candidate_keys_dict = dict()
    for val in candidate_keys:
        candidate_keys_dict[val]= True
            
    params={
        'list_categorical_attributes': categorical_attributes,
        'category_threshold': config.data_synthesizer['threshold_value'],
        'candidate_keys': candidate_keys_dict, 
        'histogram_bins': config.data_synthesizer['histogram_bins'],
        'categorical_attributes': cat_attributes # TO DO
    }

    num_of_rounds = config.epochs //  config.genetic_algorithm['aggregation_interval']
    
    
    strategy = FedGA(
                    min_available_clients=number_of_clients,
                    min_evaluate_clients=number_of_clients,
                    min_fit_clients=number_of_clients,
                    genetic_algorithm_parameters=config.genetic_algorithm,
                    total_number_of_rounds=num_of_rounds,
                    aggregation_type = config.aggregation_type,
                    threshold = config.threshold,
                    seed = seed
                    )
    temp_dir=save_dir
    
        
    # Check if path for saving results exists, if not creates the corresponding folders
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    
    client_resources = {'num_cpus':1, 'num_gpus':0}

    
    ray.init(object_store_memory=1_000_000_000,ignore_reinit_error=True)
        
    # Start federated simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(clients_partition,train_data,temp_dir,**params),
        num_clients=number_of_clients,
        config=fl.server.ServerConfig(num_rounds=num_of_rounds,round_timeout=2000),
        client_resources=client_resources,
        strategy=strategy,
        ray_init_args={"runtime_env":{"py_modules": py_modules}},   
    )
    
    ray.shutdown()
    
    #Define the paths to save the global model and the synthetic data from the federated simulation
    description_file_fed_ga = os.path.join(temp_dir, f"ga_corrmeta_fed.json")
    
    synthetic_data_fed= generate_synthetic_data(description_file_fed_ga,train_data.shape[0],seed)

    
     
    return synthetic_data_fed

 
def analysis_parameters_sensitivity():
    """ Analysis parameters of the federated training using Weight&Biases sweep"""
    
    # Start run in wandb 
    run=wandb.init(project="test")
    config=wandb.config
    
    py_modules=["DataSynthesizer","flower"]
    
    data= pd.read_csv(config.data_path)

    target=config.target
    seed=config.seed
    
    np.random.seed(seed)
    
    # First load the data and split the data into training and testing data. 
    splits_experiment = repeated_holdout(data,1,test_size=0.5) 
    
    run_id=run.id
    
    config=wandb.config
    print(config.data_synthesizer)
    
    # Initialize variable to save the results
    scores=[]
    sdv_scores=[]
    
    # Federated configuration
    number_of_clients=config.federated_learning['number_of_clients']
    scenario=config.federated_learning['scenario']
    
    results =  pd.DataFrame()
    
    for experiment_idx, (train_idx, test_idx ) in splits_experiment.items():
        # Hold-out split 
        train_data = data[data.index.isin(train_idx)].reset_index(drop=True).copy(deep=True)
        test_data = data[data.index.isin(test_idx)].reset_index(drop=True).copy(deep=True)
        
        # Partition training data into the number of clients in the federated experiment based on the scenario
        clients_partition =  partition_dataset_pandas(train_data, number_clients=number_of_clients, scenario=scenario, target=target, seed=seed)
        print(clients_partition)
        
        # Plot distribution scenario in the federated learning scenario.
        fig=plot_partition_plotly(clients_partition, train_data[target].values)
        fig.show()
        file_path= os.path.abspath(os.curdir)
        # Create save dir for the experiment
        output_dir = f'{file_path}/results/{config.dataset_name}/parameters/{run_id}'
        
        # Check if path for saving results exists, if not creates the corresponding folders
        if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        else:
                shutil.rmtree(output_dir) # Remove existing result from previous experiment with same specifications
                os.makedirs(output_dir)

        # Launch federated trainng for the given split 
        start= time.time()
        synthetic_data = federated_training(config, train_data, py_modules, number_of_clients=number_of_clients,clients_partition=clients_partition,save_dir=output_dir)
        end = time.time()
        wandb.log({f"time_federated_training_{experiment_idx}": end-start})
        
        # Set numerical columns
        numerical_columns=[ col for col in data.columns if col not in list(config.data_synthesizer["categorical_attributes"])]
        
        # Evaluate the synthetic data score
        start= time.time()
        #results=evaluate_multiple_dimension_synthetic_data(train_data,synthetic_data,list(config.data_synthesizer["categorical_attributes"]),numerical_columns,output_dir,path_metrics=None)
        all_results=pd.DataFrame()
        for client_values in clients_partition.values():
            local_data = train_data.filter(items = client_values, axis=0)
            #indexes_sample_synthetic_data = np.random.choice(synthetic_data.index,replace=False,size=local_data.shape[0])
            #sample_synthetic_data = synthetic_data.filter(items = indexes_sample_synthetic_data, axis=0)
            evaluator = sg.Evaluator(local_data, synthetic_data)
            evaluator.add_custom_metric('log_cluster',log_cluster_metric,categorical_columns=list(config.data_synthesizer["categorical_attributes"]))
            evaluator.add_custom_metric('sdv_fidelity',sdv_metrics)
            evaluator.add_custom_metric('average_hellinger_distance',average_hellinger_distance,categorical_columns=list(config.data_synthesizer["categorical_attributes"]))
            evaluator.add_custom_metric('pairwise_correlation_difference',pairwise_correlation_difference,categorical_columns=list(config.data_synthesizer["categorical_attributes"]),plot=False,save_dir=output_dir)
            evaluator.add_custom_metric('distance_to_closest_record',distance_to_closest_record, categorical_columns=list(config.data_synthesizer["categorical_attributes"]),numerical_columns=numerical_columns)
            results=evaluator.evaluate(as_df=True)
            all_results=pd.concat([results.transpose(),all_results])
        
        all_results = all_results.mean(axis=0).transpose()
        print(all_results)   
            
        for metric in all_results.index:
            wandb.log({f"{metric}":round(all_results.loc[metric],4)})
        score = round(all_results.iloc[0],3)
        sdv_score = round(all_results.iloc[3],3)
        print('This is the obtained scored:')
        print(results)
        print(score)
        print(sdv_score)
        end = time.time()
        scores.append(score)
        sdv_scores.append(sdv_score)
        
    avg_lc = np.mean(scores) 
    avd_sdv_score = np.mean(sdv_scores)
    wandb.log({"sdv_fidelity-sdv_score": avd_sdv_score})
    wandb.log({"avg_log_cluster": avg_lc})
    
    

def main():    
    parser = argparse.ArgumentParser(description="Flower Client")
    
    parser.add_argument(
        "-cf",
        "--config_file",
        type=str,
        required=True,
        help="Specifies the configuration file for the experiment",
    ) 
    
    args = parser.parse_args()
    
    # Parse config file from yaml
    config_dict = load_yaml_to_dict(args.config_file)
    print(config_dict["sweep_configuration"])
    
    # For hyperparameter tuning 
    sweep_id = wandb.sweep(sweep=config_dict["sweep_configuration"], project="test")
    wandb.agent(sweep_id, function=analysis_parameters_sensitivity)
    
if __name__ == "__main__":
    main() 