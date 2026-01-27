import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sdnist.metrics.propensity import PropensityMSE
from evaluation.utils import encode_categorical_columns, preprocess_dataset, scale_numerical_columns

def log_cluster_metric(real_data,synth_data,categorical_columns,max_size_cluster=20):
    """
    Calculates the log-cluster metric between the synthetic and real data, which indicates the similarity of the latent structure of the real 
    and synthetic data based on clustering. Larger values indicate poor similarity and viceversa. 
    
    Parameters
    -----------
    real_data : pandas.DataFrame
        Dataframe containing the real data.
    synth_data : pandas.DataFrame
        Dataframe containing the synthetic data.
    categorical_columns: list
        List of categorical attributes. 
    max_size_cluster: int
        Maximum size of clusters to consider.     
    
    Returns
    -----------
    log_cluster: pd.DataFrame
        Dataframe with mean and std values from the predictions obtained for the target value given the combinations of quasi-identifiers.
    """
    
    # Drop missing values
    real_data = real_data.dropna()
    synth_data = synth_data.dropna()
    
    # Combine
    real_synth=pd.concat([real_data,synth_data])
    labels_real=np.zeros(real_data.shape[0])
    labels_synth=np.ones(synth_data.shape[0])
    labels=np.concatenate((labels_real,labels_synth),axis=0)
    
    # Preprocess
    real_synth = preprocess_dataset(real_synth,categorical_columns)
    
    # We use PCA to reduce the dimensionality of the data.
    pca = PCA(n_components=None)
    pca.fit(real_synth)
    
    for i in range(1,len(pca.explained_variance_ratio_)):
        if sum(pca.explained_variance_ratio_[0:i])>0.85:
            break
    optimal_components=i
    
    pca = PCA(n_components=optimal_components)
    tranformed_real_synth = pca.fit_transform(real_synth)
    
    c = len(labels_real)/len(labels)
    
    # We check for the different clusters the minimum log likelihood.
    log_cluster_metric_per_cluster=[]
    for cluster_num in range(2,max_size_cluster):
        model = KMeans(n_clusters=cluster_num,random_state=42).fit(tranformed_real_synth)
        clusters = model.labels_.tolist()
        results=pd.DataFrame()
        results['cluster_num']=clusters
        results['source']=labels
        cluster_score_sum = 0
        for label in range(cluster_num):
            ratio = results.query(f'source == 0 and cluster_num == {label}').shape[0]/results.query(f'cluster_num == {label}').shape[0]
            cluster_score_sum += (ratio - c)**2
        log_cluster_metric = math.log2(cluster_score_sum/cluster_num)
        log_cluster_metric_per_cluster.append(log_cluster_metric)
        
    return min(log_cluster_metric_per_cluster)
    
    
def propensity_score_nist(real_data,synth_data,categorical_columns,numerical_columns,save_dir):
    """
    Calculates the propensitx mean squeare metric using the nist tool.
    
    Parameters
    -----------
    real_data : pandas.DataFrame
        Dataframe containing the real data.
    synth_data : pandas.DataFrame
        Dataframe containing the synthetic data.
    categorical_columns: list
        List of categorical attributes. 
        
    Returns
    -----------
    pmse: int 
        Propensity Mean Square Error
    """
    real_data, synth_data = scale_numerical_columns(real_data, synth_data, numerical_columns=numerical_columns)
    real_data,synth_data = encode_categorical_columns(real_data, synth_data, categorical_columns=categorical_columns)
    psme=PropensityMSE(real_data,synth_data,Path(save_dir)).compute_score()
    return psme