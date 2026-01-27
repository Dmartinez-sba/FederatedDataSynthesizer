# Federated DataSynthesizer

This is an extended version of the Enhanced Data Synthesizer [EnhancesDataSynthesizer](https://github.com/sbaresearch/EnhancedDataSynthesizer), which generates synthetic data that simulates a given
dataset integrating GA algorithms and mitigation strategies against attribute disclosure. For details on that project and usage, please refer to the official website!

This extended version presents a federated extension of the Enhanced Data Synthesizer, in which the migration of individuals across multiple clients at each federated round of the genetic algorithm is used to enhance diversity in Bayesian network structure learning and to learn a global model capable of generating synthetic data that follows the joint distribution of all clients.

The repository is organized as follows:
 - models: Contains the code for running federated, centralized and local training of BN model.
 - experiments_config: Contains YAML files for specifying model hyperparameters and experiment configuration values used in the experiments.
 - flower: Contains the client and server code for federated training using the flower framework.
 - evaluation: Includes all the metrics used for fidelity, utility and privacy evaluation of synthetic data. 

# Running the code 

To run the implemented experiments, follow the steps below:

1. Create a virtual environment with Python 3.9:
```bash
conda create --prefix ./.venv python=3.9 -y
```

2. Activate the environment
```bash
conda activate ./.venv 
```

3. Install the required dependencies using the following command:
```bash
pip install -r requirements_minimal.txt 
```

4. Then, run the following command for a set of experiments: 
```bash
   python main.py -cf experiments_config/adult_experiments_tuned -dn adult -fr results -ef epsilon0
```

# Federated Setup 

The input data is first split into a training set (80%) and a test set (20%). The training set is then partitioned among the number of clients specified in the YAML configuration file, following one of three distribution types selected in the configuration: 'iid', 'unbalance', or 'noniid-labeldir' distribution. 

# Baselines (Centralized, Local)

The federated experiments, are compared with two baselines: 

- Centralized baseline: This case generates synthetic data from all the centralized training data using the Enhanced Data Synthsizer original code. 
- Local baseline: This case generates synthetic data from the local data at each of the clients in the federated setup using the Enhanced Data Synthsizer original code. To compare the results to the federated and centralized baseline, average metrics for this case are reported. 

# Evaluation part 
All experiments are conducted three times to ensure robustness, with results reported as the mean and standard deviation of the evaluation metrics. 
