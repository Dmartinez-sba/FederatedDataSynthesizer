import argparse
import os 
from pathlib import Path
import sys
from typing import Dict

sys.path.append('.') # to import local DataSynthesizer 

from hydra import initialize, compose, initialize_config_dir
from omegaconf import OmegaConf
import shutil
import pickle
import time

import numpy as np
import pandas as pd
import flwr as fl
import plotly.express as px
import plotly.io as pio   

from evaluation.metrics.privacy_attacks import attribute_disclosure_as_ml_task, compute_mean_attribute_disclosure
from DataSynthesizer.federated_utils.partitions import repeated_holdout, plot_partition_plotly, partition_dataset_pandas
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from evaluation.evaluation import evaluate_multiple_dimension_synthetic_data, generate_average_report
from evaluation.plots import plot_distance_closest_record_distribution, plot_univariate_distributions
from evaluation.metrics.utility import machine_learning_evaluation, generate_average_utility_report

from models.train import centralized_training, federated_training, local_training
import os

def check_files_exist(file_paths):
    for file_path in file_paths:
        if not os.path.exists(file_path):
            return False
    return True


def main(config,output_dir):
    
    # Calculate time to run experiment
    start = time.time()
    
    # Read data
    data = pd.read_csv(config.data_path)
    target=config.target
    seed=config.seed
    dataset_name= config.dataset_name
    
    
    # Federated configuration
    number_of_clients=config.federated_learning['number_of_clients']
    scenario=config.federated_learning['scenario']
    aggregation=config.aggregation_type
    epsilon=config.genetic_algorithm.epsilon
        
    # Experiment config
    n_splits=config.n_splits

    
    # Create save dir for the experiment
    output_dir_cen = f'{output_dir}/centralized_results/'
    output_dir = f'{output_dir}/federated_results/experiment_{scenario}_{number_of_clients}_clients'
    
    
    # Check if path for saving results exists, if not creates the corresponding folders
    if not os.path.exists(output_dir_cen):
            os.makedirs(output_dir_cen)
    
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    
    # First load the data and split the data into training and testing data. 
    splits_experiment = repeated_holdout(data,n_splits) 
    
    # Initialize results for the different settings (Local, Central, Federated)
    results=pd.DataFrame()
    other_results=pd.DataFrame()
    
    attribute_disclosure_fed_results = {}
    attribute_disclosure_cen_results = {}
    
    # Initialize results for attribute disclosure if available in experiment
    if 'attribute_disclosure' in config:
        
        for i in range(config.attribute_disclosure.number_of_scenarios):
            attribute_disclosure_fed_results[i] = []
            attribute_disclosure_cen_results[i] = []
        
    
    # Set py modules to run in flower
    py_modules=["./DataSynthesizer/","./flower"]
    
    # Set metrics
    metrics=['accuracy','balanced_accuracy','auc','f1']
    
    # Read categorical columns
    categorical_columns=list(config.data_synthesizer.categorical_attributes)
    numerical_columns=[ col for col in data.columns if col not in categorical_columns]

    # Set target labels
    target_labels=data[target].unique()
    
    
    # Comparison scenarios
    scenarios=['fed','cen']
    
    # Initialize the experiment and repeat n times. 
    for experiment_idx, (train_idx,test_idx) in splits_experiment.items():
        
        # save dir fed and loc
        save_dir = f'{output_dir}/{experiment_idx}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # save dir cen 
        save_dir_cen = f'{output_dir_cen}/{experiment_idx}'
        if not os.path.exists(save_dir_cen):
            os.makedirs(save_dir_cen)
        
        # Hold-out split 
        train_data = data[data.index.isin(train_idx)].reset_index(drop=True).copy(deep=True)
        test_data  = data[data.index.isin(test_idx)].reset_index(drop=True).copy(deep=True)
        
        
        # Partition training data into the number of clients in the federated experiment based on the scenario
        clients_partition =  partition_dataset_pandas(train_data, number_clients=number_of_clients, scenario=scenario, target=target, seed=seed)
        
        
        # Run scenarios
        start_all_scenarios=time.time()
        syn_file = Path(f'{save_dir}/synthetic_data_fed.pkl')
        if syn_file.is_file():
            synthetic_data_fed=pd.read_pickle(syn_file)
        else:
            synthetic_data_fed = federated_training(config,train_data,py_modules,number_of_clients,clients_partition,save_dir=save_dir)
            synthetic_data_fed.to_pickle(f'{save_dir}/synthetic_data_fed.pkl')
            if experiment_idx == 0: 
                plot_univariate_distributions(train_data,synthetic_data_fed,categorical_columns,save_dir=save_dir)
            
        
        syn_file = Path(f'{save_dir_cen}/synthetic_data_cen.pkl')
        if syn_file.is_file():
            synthetic_data_cen=pd.read_pickle(syn_file)
        else:
            synthetic_data_cen= centralized_training(config,train_data,seed=seed,save_dir=save_dir_cen)
            synthetic_data_cen.to_pickle(syn_file)
            if experiment_idx == 0: 
                plot_univariate_distributions(train_data,synthetic_data_cen,categorical_columns,save_dir=save_dir_cen)
        
        
        if config.local:
            file_list_local=[os.path.join(save_dir, f"synthetic_local{i}.pkl") for i in range(0,number_of_clients)]
            if check_files_exist(file_list_local):
                synthetic_datasets_local = [pd.read_pickle(file) for file in file_list_local]
                print('Entered local synthetic datasets')
            else:
                synthetic_datasets_local = local_training(config,train_data,clients_partition,save_dir=save_dir) 
                for i,synthetic_df in enumerate(synthetic_datasets_local):
                    synthetic_df.to_pickle(os.path.join(save_dir, f"synthetic_local{i}.pkl"))
            scenarios=['fed','cen','loc']
            
        end_all_scenarios=time.time()
        print(f' Time all scenarios {end_all_scenarios-start_all_scenarios}')
        
        # ML performance original data
        results_original= machine_learning_evaluation(train_data,test_data,target,target_labels,metrics,f'real_{experiment_idx}')
        print('machine learning results done')
        
        # Evaluate the corresponding synthetic data (in terms of Utility)
        results_fed_synthetic= machine_learning_evaluation(synthetic_data_fed,test_data,target,target_labels,metrics,f'synthfed_{experiment_idx}')
        results_cen_synthetic= machine_learning_evaluation(synthetic_data_cen,test_data,target,target_labels,metrics,f'synthcen_{experiment_idx}')
        print('Utility Evaluation Done!')

        # Evaluate synthetic data (in terms of Fidelity and Privacy)
        results_add_fed_synthetic=evaluate_multiple_dimension_synthetic_data(train_data,synthetic_data_fed,categorical_columns,numerical_columns,save_dir=save_dir,plot=True).add_suffix(f'_syn_fed_{experiment_idx}')
        results_add_cen_synthetic=evaluate_multiple_dimension_synthetic_data(train_data,synthetic_data_cen,categorical_columns,numerical_columns,save_dir=save_dir,path_metrics=None).add_suffix(f'_syn_cen_{experiment_idx}')
        print('Evaluation fidelity part one done')
        plot_distance_closest_record_distribution(train_data,synthetic_data_fed,test_data,categorical_columns, numerical_columns, output_dir)
        plot_distance_closest_record_distribution(train_data,synthetic_data_cen,test_data,categorical_columns, numerical_columns, output_dir_cen)
        
        print('Fidelity Evaluation Done!')
        
        # Evaluate federated synthetic data in terms of attribute disclosure (centralized and federated)
        if 'attribute_disclosure' in config:
            for i in range(config.attribute_disclosure.number_of_scenarios):
                quasi_identifiers= list(config.attribute_disclosure[f'scenario{i}'].quasi_identifiers)
                target_attr_discl = config.attribute_disclosure[f'scenario{i}'].target
                key_length = config.attribute_disclosure[f'scenario{i}'].key_length
                attr_disc = attribute_disclosure_as_ml_task(train_data,synthetic_data_fed,quasi_identifiers,key_length,target_attr_discl,categorical_columns)
                print(attr_disc)
                attribute_disclosure_fed_results[i].append(attr_disc)
                attr_disc = attribute_disclosure_as_ml_task(train_data,synthetic_data_cen,quasi_identifiers,key_length,target_attr_discl,categorical_columns)
                attribute_disclosure_cen_results[i].append(attr_disc)
            print('Attribute Disclosure Done!')
        
        if other_results.empty:
            other_results=pd.concat([other_results,results_add_fed_synthetic.join(results_add_cen_synthetic)])
        else:
            other_results=other_results.join(results_add_fed_synthetic)
            other_results=other_results.join(results_add_cen_synthetic)
        
        if config.local:
            for i,synth_data_loc in enumerate(synthetic_datasets_local):
                pass
                results_loc_synthetic= machine_learning_evaluation(synth_data_loc,test_data,target,target_labels,metrics,f'synthloc_{experiment_idx}_client{i}')
                results_add_loc_synthetic=evaluate_multiple_dimension_synthetic_data(train_data,synth_data_loc,categorical_columns,numerical_columns,save_dir=save_dir,path_metrics=None).add_suffix(f'_syn_loc_{experiment_idx}_{i}')
                if experiment_idx == 0: 
                    plot_univariate_distributions(train_data,synth_data_loc,categorical_columns,save_dir=f'{save_dir}/local/client_{i}')
                results=pd.concat([results,results_loc_synthetic])
                other_results=other_results.join(results_add_loc_synthetic)
        
        # Concatenate the results obtained
        results=pd.concat([results,results_original,results_cen_synthetic,results_fed_synthetic])   
        
        # Save intermediate results
        results.to_csv(os.path.join(output_dir, f"results_utility_baseline_{experiment_idx}.csv"),index=True)
        other_results.transpose().to_csv(os.path.join(output_dir, f"results_fidelity_privacy_baseline_{experiment_idx}.csv"),index=True)
           
    # Generate reports with the results obtained 
    total_results=generate_average_utility_report(results,metrics=metrics,n_splits=n_splits,scenarios=scenarios)
    total_other_results=generate_average_report(other_results,n_splits=n_splits ,scenarios=scenarios)
    total_results.to_csv(os.path.join(output_dir, f"results_utility_baseline.csv"),index=True)
    total_other_results.transpose().to_csv(os.path.join(output_dir, f"results_fidelity_privacy_baseline.csv"),index=True)
    print('Finish correctly all utility and fidelity')
    if attribute_disclosure_fed_results != {}:
        compute_mean_attribute_disclosure(attribute_disclosure_fed_results,save_dir=output_dir)
        compute_mean_attribute_disclosure(attribute_disclosure_cen_results,save_dir=output_dir_cen)
        
    end = time.time()
    print(f'Total time of the experiment: {end - start}')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Flower Client")
    
    parser.add_argument(
        "-cf",
        "--config_file",
        type=str,
        required=True,
        help="Specifies the configuration file for the experiment",
    ) 
    
    parser.add_argument(
        "-dn",
        "--dataset_name",
        type=str,
        required=True,
        help="Specifies the name of the dataset",
    ) 
    
    parser.add_argument(
        "-fr",
        "--folder_results",
        type=str,
        required=True,
        help="Specifies the name of the folder for results",
    ) 
    
    parser.add_argument(
        "-ef",
        "--epsilon_folder",
        type=str,
        required=True,
        help="Specifies the epsilon folder of experiments",
    )
    
       
    args = parser.parse_args()
    dataset_name=args.dataset_name
    
    config_file_path=args.config_file
    results_file_path=args.folder_results
    epsilon_folder=args.epsilon_folder

    
    for folder in os.listdir(config_file_path):
        internal_folder=f'{config_file_path}/{folder}'
        for file in os.listdir(internal_folder):
            if file.endswith(".yaml"):
                    with initialize(config_path=internal_folder, version_base=None):
                            config = compose(config_name=file)
                            try:
                                print(f'{results_file_path}/{dataset_name}/{folder}')
                                if (folder.split('/')[-1] == epsilon_folder):
                                    print(f'{results_file_path}/{dataset_name}/{folder}')
                                    main(config,f'{results_file_path}/{dataset_name}/{folder}')
                                    print('Finish Succesfully!')
                            except Exception as X:
                                print(X)
                                print(f'Error with file {file}') 

    