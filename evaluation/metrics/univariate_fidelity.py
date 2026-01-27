from collections import namedtuple
import numpy as np
import pandas as pd
from scipy.linalg import norm
from scipy.spatial import distance
from evaluation.utils import encode_categorical_columns
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import SingleTableMetadata

def _get_probability_vector(real_data, synthetic_data, feature, categorical):
    """ 
    Determines the probability distribution of a feature in the real and synthetic data. 
    
    Parameters
    ----------
    real_data : pandas.DataFrame
        Dataframe containing the real data.
    synth_data : pandas.DataFrame
        Dataframe containing the synthetic data.
    feature : str
        Feature to process.
    categorical : bool
        Indicates if a feature is categorical.
        
    Returns
    -------
    p_real, p_synth : np.ndarray
        The probability vectors for the real and synthetic data for the given feature.
    """
    
    real_values = real_data[feature].dropna()
    synth_values = synthetic_data[feature].dropna()
    combined=pd.concat([real_values,synth_values])
    
    if categorical:
        unique=len(combined.unique())
        bin_edges=np.histogram_bin_edges(real_values, bins=unique, range=(0, unique))
    else:
        bin_edges = np.histogram_bin_edges(combined, bins='auto')
    
    p_real = np.histogram(real_values, bin_edges)[0]/len(real_values)
    p_synth = np.histogram(synth_values, bin_edges)[0]/len(synth_values)

    return p_real, p_synth

def _hellinger_function(p, q):
    """ 
    Calculates the hellinger distance between two discrete probability distributions.
    
    Parameters
    ----------
    p : np.array
        Probability vector.
    q : np.array
        Probability vector.

    Returns
    -------
    hellinger_distance: int
        Distance values between 0 and 1.
    """
    return norm(np.sqrt(p)-np.sqrt(q))/np.sqrt(2)

def hellinger_distance(real, synth, categorical_columns=[]):
    """ 
    Calculates the hellinger distance between each feature of the real and synthetic data.
    
    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data. respectively.
    categorical_columns: list
        List of categorical attributes. 

    Returns
    -------
    hellinger_distances: List
        List with distance values between 0 and 1.
    """
    real,synth = encode_categorical_columns(real,synth,categorical_columns)
    hellinger_distances=[]
    for column in real.columns:
        categorical = column in categorical_columns
        p_real,p_synth=_get_probability_vector(real,synth,column,categorical)   
        hellinger_distances.append(round(_hellinger_function(p_real,p_synth),4)) 
    return hellinger_distances

# add wasserstein distance for numeric attributes
 


def average_hellinger_distance(real, synth, categorical_columns=[]):
    """ 
    Calculates the average hellinger distance between features in the real and synthetic data.
    
    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data. respectively.
    categorical_columns: list
        List of categorical attributes. 

    Returns
    -------
    avg_hellinger_distance: int
        Distance value between 0 and 1.
    """
    avg_hellinger_distance=np.mean(hellinger_distance(real,synth,categorical_columns))
    return avg_hellinger_distance

def jensen_shannon_distance(real, synth, categorical_columns=[]):
    """ 
    Calculates the jesen shannon distance between two discrete probability distributions.
    
    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.

    Returns
    -------
    js_distance: int
        Distance value between 0 and 1.
    """
    real,synth = encode_categorical_columns(real,synth,categorical_columns)
    js_distances=[]
    for column in real.columns:
        categorical = column in categorical_columns
        p_real,p_synth=_get_probability_vector(real,synth,column,categorical) 
        distance_js= distance.jensenshannon(p_real,p_synth)
        js_distances.append(round(distance_js,4))
    return js_distances

def average_jensen_shannon_distance(real, synth, categorical_columns=[]):
    """ 
    Calculates the average jensen-shannon distance between features in the real and synthetic data.
    
    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data. respectively.
    categorical_columns: list
        List of categorical attributes. 

    Returns
    -------
    avg_js_distance: int
        Distance value between 0 and 1.
    """
    avg_js_distance=np.mean(jensen_shannon_distance(real,synth,categorical_columns))
    return avg_js_distance

def sdv_metrics(real, synth):
    """ 
    Provides a quality score of the synthetic data based on different metrics. 
    For more information, refer to https://docs.sdv.dev/sdmetrics/reports/quality-report/single-table-api
    
    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data. respectively.

    Returns
    -------
    sdv_shapes : float
        Score SDV for univariate distributions similarity.
    sdv_pair_trends : float
        Score SDV for pair trends similarity.
    std_dcr : float
        Overall score SDV for quality of synthetic data.
    """
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real)
    
    quality_report = evaluate_quality(
                        real_data=real,
                        synthetic_data=synth,
                        metadata=metadata)
    
    results=quality_report.get_properties()
    
    properties =  results['Property'].values
    
    new_entry = pd.DataFrame([{'Property':'Overall Score', 'Score':round(quality_report.get_score(),5)}])
    results= pd.concat([results,new_entry], ignore_index=True)
    
    scores=results['Score'].values
        
    SDVResults = namedtuple(
        "SDVResults",
        ("sdv_shapes","sdv_pair_trends","sdv_score"),
    )
    return SDVResults(scores[0],scores[1],scores[2])
    