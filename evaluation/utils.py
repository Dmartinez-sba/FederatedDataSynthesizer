import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import warnings
from scipy.stats import ttest_ind

def encode_categorical_columns(real_data,synth_data,categorical_columns):
    """ Encodes categorical columns in both real and synthetic data. 

    Parameters
    ----------
    real_data : pandas.DataFrame
        Dataframe containing the real data.
    synth_data : pandas.DataFrame
        Dataframe containing the synthetic data.
    categorical_columns : List
        List with categorical features.

    Returns
    -------
    real, synth : pandas.DataFrame
        Real and synthetic dataframe encoded.  
    
    """
    real,synth = real_data.copy(),synth_data.copy()
    combined_data = pd.concat([real, synth], axis=0)
    
    for col in categorical_columns:
        combined_data[col] = combined_data[col].astype('category').cat.codes
    
    real_transformed = combined_data.iloc[:real.shape[0]]
    synthetic_transformed = combined_data.iloc[real.shape[0]:]
    
    return real_transformed,synthetic_transformed

def scale_numerical_columns(real_data,synth_data,numerical_columns):
    """ Scales categorical columns in both real and synthetic data. 

    Parameters
    ----------
    real_data : pandas.DataFrame
        Dataframe containing the real data.
    synth_data : pandas.DataFrame
        Dataframe containing the synthetic data.
    categorical_columns : List
        List with categorical features.

    Returns
    -------
    real, synth : pandas.DataFrame
        Real and synthetic dataframe encoded.  
    
    """
    real,synth = real_data.copy(),synth_data.copy()
    combined_data = pd.concat([real, synth], axis=0)
    
    scaler = MinMaxScaler()
    combined_data[numerical_columns]=scaler.fit_transform(combined_data[numerical_columns].to_numpy())
    
    real_transformed = combined_data.iloc[:real.shape[0]]
    synthetic_transformed = combined_data.iloc[real.shape[0]:]
    
    return real_transformed,synthetic_transformed



def majority_voting(predictions_list):
    """ Performs mayority voting from a vector containing predictions from different classifiers. 
    
    Parameters
    ----------
    predictions_list : list
        List containing prediction vectors of different classifiers.
    Returns
    -------
    ensemble_result: list
        List containing predictions based on majority voting.
    
    """
    ensemble_result=[]
    for i in range(len(predictions_list[0])):
        predictions=[ val[i] for val in predictions_list]
        most_frequent_val = most_frequent(predictions)
        ensemble_result.append(most_frequent_val)
        
    return ensemble_result

def average_results(predictions_list):
    """ Performs simple average from a vector containing predictions from different regression models. 
    
    Parameters
    ----------
    predictions_list : list
        List containing prediction vectors of different regression models.
    Returns
    -------
    ensemble_result: list
        List containing predictions based on simple average.
    
    """
    ensemble_result=[]
    for i in range(len(predictions_list[0])):
        predictions=[ val[i] for val in predictions_list]
        avg_val = np.mean(predictions)
        ensemble_result.append(avg_val)
        
    return ensemble_result

    
    
def most_frequent(entry_list):
    """
    Counts more frequent element in a list.
    """
    return max(set(entry_list), key=entry_list.count) 

def preprocess_dataset(df,categorical_columns=[]):
    """ Preprocess a dataset using Ordinal Encoding for categorical features and Min-Max for numerical features. 
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to preprocess.
    Returns
    -------
    preprocess_df: pd.DataFrame
        Preprocess Data.
    """
    
    numerical_columns= [ col for col in df.columns if col not in categorical_columns]
    
    num_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])

    # Preprocess categorical columns
    cat_transformer = Pipeline(steps=[
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # Build transformer with preprocessing pipelines.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", num_transformer, numerical_columns),
            ("categorical", cat_transformer, categorical_columns),
        ]
    )   
    
    df_preprocess=pd.DataFrame(preprocessor.fit_transform(df))
    df_preprocess.columns = numerical_columns + categorical_columns
    return df_preprocess


def handle_warnings_as_exceptions():
    warnings.filterwarnings('error', category=RuntimeWarning)

def t_test_color(val1, val2, alpha=0.05, decimal_places=5):
    '''
    T-test for two sample groups val1 and val2. 
    
    Parameters
    ----------
    val1 : list
        List with means of baseline1.
    val2 : list
        List with means of baseline2.
    alpha: float
        Threshold for p-value.
    decimal_places: int
        Rounding 
    Returns
    -------
    df_significance_testing : pandas.DataFrame
        Dataframe colored based on the significance of models against reference.  
        
    '''
    val1 = np.round(val1, decimals=decimal_places)
    val2 = np.round(val2, decimals=decimal_places)
    # Compute basic statistics
    handle_warnings_as_exceptions()
    try:
        t_stat, p_val = ttest_ind(val1, val2, equal_var=True)
    except RuntimeWarning as e:
        return 'white'
    mean1, mean2 = np.mean(val1), np.mean(val2)
    if p_val < alpha:
        if mean2 < mean1:
            return 'green'  # Federated/Local is better (smaller)
        else:
            return 'red'  # Federated/Local is worse (larger)
    return 'white'
    
def significance_testing(dataset_name,path_folder_results, metrics, rows, abbr_metrics):
    '''
    Test significance for different metrics comparing federated baseline against centralized baseline and 
    local baseline against federated baseline. 
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    path_folder_results : str
        Path containing the folder with all results.
    metrics : list
        List with all the metrics that are compared for each baseline.
    rows : list
        List with all the scenarios considered.
    abbr_metrics: Dict
        Abbreviation of the metrics to write in the dataframe.

    Returns
    -------
    df_significance_testing : pandas.DataFrame
        Dataframe colored based on the significance of models against reference.  
    
    '''
    df_significance_testing=pd.DataFrame()
    epsilon_values=[0,0.5,1.6]
    for epsilon_value in epsilon_values:
        for row in rows:
            results=pd.read_csv(f'{path_folder_results}/{dataset_name}/epsilon{epsilon_value}/federated_results/experiment_{row}_clients/results_fidelity_privacy_baseline_2.csv')
            for metric in metrics:
                fed_values = results[results['Unnamed: 0'].str.contains('fed')][metric].values
                cen_values = results[results['Unnamed: 0'].str.contains('cen')][metric].values
                local_values = [np.mean(results[results['Unnamed: 0'].str.contains(f'loc_{experiment_idx}')][metric].values) for experiment_idx in range(3)]
                sig_testing_cen_fed = t_test_color(cen_values,fed_values,alpha=0.05,scenario=f'{metric}_eps{epsilon_value}_cen_fed')
                sig_testing_fed_loc = t_test_color(fed_values,local_values,alpha=0.05,scenario=f'{metric}_eps{epsilon_value}_fed_loc')
                df_significance_testing.at[f'{row}_fed',f'{abbr_metrics[metric]}_eps{epsilon_value}']=sig_testing_cen_fed
                df_significance_testing.at[f'{row}_loc',f'{abbr_metrics[metric]}_eps{epsilon_value}']=sig_testing_fed_loc
    def apply_color(val):
        return f"background-color: {val}"
    return df_significance_testing.style.applymap(apply_color)