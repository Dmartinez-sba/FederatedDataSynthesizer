import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from fedlab.utils.dataset.functional import balance_split,homo_partition,dirichlet_unbalance_split, hetero_dir_partition

def partition_dataset( file_path, number_clients, scenario, target, alpha=0.5, min_require_size=200, seed=0):
    '''
    Partitions a dataset in multiple clients using different distribution strategies (based on FedLab). 
        
    Arguments:
        file_path (str): Absolute path of dataset to partition.
        number_clients (int): Total number of clients.
        scenario (str): Partition type. Supported values (iid, unbalance and noniid-labeldir).
        target (str): Target variable.
        alpha (float, optional): Parameter for Dirichlet distribution (only for unbalance and noniid-labeldir partition).
        min_require_size (int, optional): Parameter to determine the minimum number of samples assigned to a client in the noniid-labeldir partition.
       
    Returns:
        dictionary: A dictionary containing the index of samples assign to each client. 
    '''
    np.random.seed(seed)
    
    format= file_path.split('.')[1]
    if format == "csv":
        dataset = pd.read_csv(file_path)  
    elif format == "pkl":
        dataset = pd.read_pickle(file_path)
    else:
        raise ValueError(f"The file format {format} is not supported.")   
    number_of_samples = dataset.shape[0]
    client_ids = np.arange(0,number_clients)
    
    le=LabelEncoder()
    targets=le.fit_transform(dataset[target].values)
    num_classes=len(dataset[target].unique())
    
    if scenario == 'iid':
        # Quantity balanced (uniform distribution)
        partition_size = number_of_samples // number_clients
        indexes = np.random.permutation(number_of_samples)
        splits = [(i+1)*partition_size for i in range(number_clients)]
        clients_indexes = np.split(indexes, splits)
        
        client_dict=dict(zip(client_ids,clients_indexes))
                         
    elif scenario == 'unbalance':      
        # Quantity-skew (Dirichlet)
        client_sample_nums = dirichlet_unbalance_split(number_clients, number_of_samples,
                                                             alpha)
        client_dict = homo_partition(client_sample_nums, number_of_samples)
    
    elif scenario == 'noniid-labeldir':
        # Label-skew (Dirichlet)
        client_dict = hetero_dir_partition(targets, number_clients, num_classes, alpha, min_require_size)
               
    else:
       raise ValueError(f" The 'scenario' {(scenario)} is not valid. The supported values for 'scenario' are 'iid' or 'non-iid'.") 
                
    return client_dict

def partition_dataset_pandas( dataset, number_clients, scenario, target, alpha=0.5, min_require_size=200, seed=0):
    '''
    Partitions a dataset in multiple clients using different distribution strategies (based on FedLab). 
        
    Arguments:
        dataset (pd.Dataframe or np.array): Dataset to partition.
        number_clients (int): Total number of clients.
        scenario (str): Partition type. Supported values (iid, unbalance and noniid-labeldir).
        target (str): Target variable.
        alpha (float, optional): Parameter for Dirichlet distribution (only for unbalance and noniid-labeldir partition).
        min_require_size (int, optional): Parameter to determine the minimum number of samples assigned to a client in the noniid-labeldir partition.
       
    Returns:
        dictionary: A dictionary containing the index of samples assign to each client. 
    '''
    np.random.seed(seed)
    
    number_of_samples = dataset.shape[0]
    client_ids = np.arange(0,number_clients)
    
    le=LabelEncoder()
    targets=le.fit_transform(dataset[target].values)
    num_classes=len(dataset[target].unique())
    
    if scenario == 'iid':
        # Quantity balanced (uniform distribution)
        partition_size = number_of_samples // number_clients
        indexes = np.random.permutation(number_of_samples)
        splits = [(i+1)*partition_size for i in range(number_clients)]
        clients_indexes = np.split(indexes, splits)
        
        client_dict=dict(zip(client_ids,clients_indexes))
                         
    elif scenario == 'unbalance':      
        # Quantity-skew (Dirichlet)
        client_sample_nums = dirichlet_unbalance_split(number_clients, number_of_samples,
                                                             alpha)
        client_dict = homo_partition(client_sample_nums, number_of_samples)
    
    elif scenario == 'noniid-labeldir':
        # Label-skew (Dirichlet)
        client_dict = hetero_dir_partition(targets, number_clients, num_classes, alpha, min_require_size)
               
    else:
       raise ValueError(f" The 'scenario' {(scenario)} is not valid. The supported values for 'scenario' are 'iid' or 'non-iid'.") 
                
    return client_dict


def plot_partition_plotly(clients_indexes, targets, ymax=None ):
    '''
    Plots the distribution of samples in each of the clients using plotly.
    
    Arguments:
        client_indexes (str): Dictionary containing the indexes of each of the clients.
        targets (np.array or list): Targets of the dataset. 
        ymax (int, optional): The top ylim in data coordinates.
        
    '''
    df=pd.DataFrame({'client':np.zeros(len(targets)),'class':targets})
    for key,val in clients_indexes.items():
        df.loc[val,'client']=key
    df['count'] = 1
    number_of_clients = len(clients_indexes.keys())
    fig = px.histogram(df, 
                   x='client', 
                   y='count', 
                   color='class',
                   height=400,
                   width=600,
                   color_discrete_sequence=px.colors.qualitative.Plotly
                   )
    fig.update_layout(
    xaxis_title="Client number",
    yaxis_title="Number of instances",
    font=dict(
        family="sans-serif",
        size=16,
        color="Black"
    ),
    xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgrey', linecolor='black', linewidth=2, ticklen=5, tickcolor='black'),
    yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgrey', linecolor='black', linewidth=2, ticklen=5, tickcolor='black'),
    legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),
    plot_bgcolor='white',
    bargap=0.5,
    )
    if ymax:
        fig.update_yaxes(tickformat=',d',range=[0,ymax])
    fig.update_xaxes(tickformat=',d', tickvals = np.arange(number_of_clients),tickmode = 'array',
                    ticktext= np.arange(number_of_clients)+1)
    return fig

def repeated_holdout(dataset, n, test_size=0.2, seed=0):
    '''
    Splits a dataset n times into training and test data using a repeated hold out method.
    
    Arguments:
        dataset (pd.dataframe or np.array): Dictionary containing the indexes of each of the clients.
        n (int): Number of splits.
        test_size (float or int): Proportion of the dataset in the test split. 
    '''
    splits_dictionary= dict()
    data_split = ShuffleSplit(n_splits=n, test_size=test_size, random_state=seed)
    for idx, (train_idx,test_idx) in enumerate(data_split.split(dataset)):
        splits_dictionary[idx] = [train_idx,test_idx]
    return splits_dictionary    
        