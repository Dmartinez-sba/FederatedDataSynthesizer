from dython.nominal import associations
import numpy as np
import warnings
warnings.filterwarnings("ignore")

 
def pairwise_correlation_difference(real,synth,categorical_columns=None,plot=False,save_dir=None):
    """ 
    Calculates the pairwise correlation difference between the real and synthetic data.
    
    Parameters
    ----------
    real (pandas.DataFrame):
        Dataframe containing the real data.
    synth (pandas.DataFrame):
        Dataframe containing the synthetic data. respectively.
    categorical_columns (list):
        List of categorical attributes. 

    Returns
    -------
    pcd (int):
        Pairwise correlation difference between correlation matrices of real and synth dataframes.
    """
    corr_real=associations(real,nominal_columns=categorical_columns,plot=plot,cmap= 'vlag',vmin=-1, vmax=1,annot=False,figsize=(10,10),filename=f'{save_dir}/real_corr.pdf')['corr']
    corr_synth=associations(synth,nominal_columns=categorical_columns,plot=plot,cmap= 'vlag',vmin=-1, vmax=1,annot=False,figsize=(10,10),filename=f'{save_dir}/syn_corr.pdf')['corr']
    pcd = np.linalg.norm(corr_real-corr_synth, 'fro')
    return pcd