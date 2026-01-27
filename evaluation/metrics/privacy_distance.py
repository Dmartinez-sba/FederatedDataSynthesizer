import numpy as np
from collections import namedtuple
from sdmetrics.single_table import NewRowSynthesis
from sdv.metadata import SingleTableMetadata
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import directed_hausdorff
from evaluation.utils import encode_categorical_columns, scale_numerical_columns
from sklearn.metrics.pairwise import cosine_similarity

##For efficiency calculations could be improved using Cython. Using metrics that support numerical and categorical variables (e.g. gower_distance).

def k_nearest_neighbor(X,Y,k,metric='euclidean'):
    """ Calculates the distances to the k nearest neighbors.

    Parameters
    ----------
        X : pandas.DataFrame
            Training data.
        Y : pandas.DataFrame 
            Query data.
        k : int
            Number of neighbors.
        metric (str, optional):
            Metric to compute the nearest_neighbors distance. Defaults to 'euclidean'.

    Returns
    -------
        distances: np.ndarray
            Distances to the k neighbors of each query record.
            
    """
    neighbors=NearestNeighbors(
        metric=metric,
        n_neighbors=k,
    ).fit(X)
    distances, _ = neighbors.kneighbors(Y)
    return distances

def exact_matches (real_data, synthetic_data, numerical_match_tolerance):
    """Determines whether each row in the synthetic data is new or matches the real data.
    For more information, refer to https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/newrowsynthesis

    Parameters
    -----------
    real_data : pandas.DataFrame
        Dataframe containing the real data.
    synth_data : pandas.DataFrame
        Dataframe containing the synthetic data.
    numerical_match_tolerance : float
        Float indicating how close two numerical values should be to considered them as match.      
        
    Returns
    -----------
    score : int
        Value between 0 and 1, representing the proportion of rows that match a row in real data.
        A value of 0 means all rows are copies and 1 there is no matching rows.
    """
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(synthetic_data)
    
    score = NewRowSynthesis.compute(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata = metadata,
    numerical_match_tolerance=numerical_match_tolerance,
    synthetic_sample_size=None 
    )
    
    return  score

def all_distances_closest_records(real_data, synthetic_data, categorical_columns, numerical_columns, metric='euclidean',k=1):
    """ Measures the distances to closest real records from each synthetic record and returns the distances.

    Parameters
    -----------
    real_data : pandas.DataFrame
        Dataframe containing the real data.
    synth_data : pandas.DataFrame
        Dataframe containing the synthetic data.
    categorical_columns : List
        List with categorical features.
        

    Returns
    -------
    distances : list
        List to closest real records.
  
    """
    
    real_data, synthetic_data = encode_categorical_columns(real_data, synthetic_data, categorical_columns=categorical_columns)
    real_data, synthetic_data = scale_numerical_columns(real_data, synthetic_data, numerical_columns=numerical_columns)
    distances = k_nearest_neighbor(real_data,synthetic_data, k=k, metric=metric)
    return distances
    
def distance_to_closest_record(real_data, synthetic_data, categorical_columns, numerical_columns, metric='euclidean'):
    """ Measures the distances to the closest real record to each synthetic record and calculates the minimum, mean and standard deviation of distances.

    Parameters
    -----------
    real_data : pandas.DataFrame
        Dataframe containing the real data.
    synth_data : pandas.DataFrame
        Dataframe containing the synthetic data.
    categorical_columns : List
        List with categorical features.
        

    Returns
    -------
        min_dcr : float
            Minimum distance to the closest record of all synthetic records.      
        mean_dcr : float
            Mean distance to the closest record of all synthetic records.
        std_dcr : float
            Standard deviation of distances to the closest record of all synthetic records.
  
    """
    
    real_data, synthetic_data = encode_categorical_columns(real_data, synthetic_data, categorical_columns=categorical_columns)
    real_data, synthetic_data = scale_numerical_columns(real_data, synthetic_data, numerical_columns=numerical_columns)
    distances = k_nearest_neighbor(real_data,synthetic_data, k=1, metric='euclidean')
    min_dcr = round(distances.min(),4) 
    mean_dcr = round(distances.mean(),4)
    std_dcr = round(distances.std(),4)
    DCRResult = namedtuple(
        "DCRResult",
        ("min_dcr", "mean_dcr", "std_dcr"),
    )
    return DCRResult(min_dcr,mean_dcr,std_dcr)

def nearest_neighbor_distance_ratio(real_data, synthetic_data, categorical_columns, numerical_columns,metric='euclidean'):
    """  Calculates the nearest neighbor distance ratio for each synthetic record by measuring the ratio of the Euclidean distances between a synthetic record 
    and its closest and second-closest neighbors in the real data. The function then computes the minimum, mean, and standard deviation of these ratios.

    Parameters
    -----------
    real_data : pandas.DataFrame
        Dataframe containing the real data.
    synth_data : pandas.DataFrame
        Dataframe containing the synthetic data.
    categorical_columns : List
        List with categorical features.
        
    Returns
    -------
        min_nndr : float
            Minimum nearest neighbor distance ratio from synthetic to real records.
        mean_nndr : float
            Mean nearest neighbor distance ratio from synthetic to real records.
        std_nndr : float
            Standard nearest neighbor distance ratio from synthetic to real records.
  
    """
    
    real_data, synthetic_data = encode_categorical_columns(real_data, synthetic_data, categorical_columns=categorical_columns)
    real_data, synthetic_data = scale_numerical_columns(real_data, synthetic_data, numerical_columns=numerical_columns)
    distances = k_nearest_neighbor(real_data,synthetic_data, k=2, metric='euclidean')
    
    nndr=np.apply_along_axis(lambda x: x[0]/x[1],1,distances)
    min_nndr = round(nndr.min(),4) 
    mean_nndr = round(nndr.mean(),4)
    std_nndr = round(nndr.std(),4)
    NNDR = namedtuple(
        "NNDRResult",
        ("min_nndr","mean_nndr", "std_nndr"),
    )
    return NNDR(min_nndr,mean_nndr,std_nndr)


def cosine_similarity(real_data,synthetic_data,categorical_columns):
    """  Calculates the pairwise cosine similarity between real and synthetic records.

    Parameters
    -----------
    real_data : pandas.DataFrame
        Dataframe containing the real data.
    synth_data : pandas.DataFrame
        Dataframe containing the synthetic data.
    categorical_columns : List
        List with categorical features.
        
    Returns
    -------
        min_cs : float
            Minimum cosine similarity between a synthetic and real record.
        mean_cs : float
            Mean cosine similarity between synthetic and real records.
        std_cs : float
            Std cosine similarity between synthetic and real records.
        max_cs : float
            Maximum cosine similarity between a synthetic and real record.
            
  
    """
    real_data, synthetic_data = encode_categorical_columns(real_data, synthetic_data, categorical_columns=categorical_columns)
    # to do min max scale numerical features
    cos_similarity = cosine_similarity(synthetic_data,real_data)
    min_cs = round(cos_similarity.min(),4) 
    mean_cs = round(cos_similarity.mean(),4)
    std_cs = round(cos_similarity.std(),4)
    max_cs = round(cos_similarity.max(),4)
    CS = namedtuple(
        "CSResult",
        ("min_cs","mean_cs", "std_cs","max_cs"),
    )
    return CS(min_cs,mean_cs,std_cs,max_cs)