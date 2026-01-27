import synthgauge as sg
import pandas as pd
from evaluation.metrics.bivariate_fidelity import pairwise_correlation_difference
from evaluation.metrics.multivariate_fidelity import log_cluster_metric, propensity_score_nist
from evaluation.metrics.privacy_distance import distance_to_closest_record, nearest_neighbor_distance_ratio
from evaluation.metrics.univariate_fidelity import average_hellinger_distance, average_jensen_shannon_distance, sdv_metrics


def evaluate_multiple_dimension_synthetic_data(real_data, synth_data,categorical_columns,numerical_columns,save_dir,plot=False,path_metrics=None):
    """ Evaluates the synthetic data in multiple dimensions (Fidelity,Privacy).

    Parameters
    ----------
    real_data : pandas.DataFrame
        Dataframe containing the real data.
    synth_data: pandas.DataFrame
        Dataframe containing the synthetic data.
    categorical_columns: list
        List of categorical attributes.
    path_metrics (optional): str
        Path to an output dir to save results.
        
    Returns
    -------
    results : pandas.DataFrame
        Dataframe with all metrics calculated from the synthetic data with respect to real data
    """
    evaluator = sg.Evaluator(real_data, synth_data,handle_nans='no')
    evaluator.add_custom_metric('average_hellinger_distance',average_hellinger_distance,categorical_columns=categorical_columns)
    evaluator.add_custom_metric('pairwise_correlation_difference',pairwise_correlation_difference,categorical_columns=categorical_columns,plot=plot,save_dir=save_dir)
    #evaluator.add_metric('propensity_metrics',method='cart',random_state=42,max_depth=6,estimator='boot')
    #evaluator.add_custom_metric('distance_to_closest_record',distance_to_closest_record, categorical_columns=categorical_columns,numerical_columns=numerical_columns)
    #evaluator.add_custom_metric('nndr',nearest_neighbor_distance_ratio, categorical_columns=categorical_columns,numerical_columns=numerical_columns)
    #evaluator.add_custom_metric('log_cluster',log_cluster_metric,categorical_columns=categorical_columns)
    #evaluator.add_custom_metric('pmse_nist',propensity_score_nist,categorical_columns=categorical_columns,numerical_columns=numerical_columns,save_dir=save_dir)
    evaluator.add_custom_metric('sdv_fidelity',sdv_metrics)
    results=evaluator.evaluate(as_df=True)
    if path_metrics != None:
        results.to_csv(f'{path_metrics}/complete_evaluation.csv')
    return results
    
def generate_average_report(results, n_splits, scenarios=['fed','cen','loc']):
    """
    Calculates the average and std of metrics of synthetic data for different baselines in multiple runs.

    Parameters
    ----------
    results : pandas.DataFrame
        Dataframe containing the evaluation results of synthetic data for multiple baselines and runs.
    n_splits : int
        Number of times the experiments were repeated.
    synth_data : list
        List containing the baselines to compare.
        
    Returns
    -------
    average_results : pandas.DataFrame
        Dataframe with metrics  of multiple rounds average and with std calculated from the synthetic data for different baselines.
    
    """
    scenarios_index=[ f'synth{scenario}' for scenario in scenarios]
    average_results=pd.DataFrame(index=scenarios_index,columns=results.index)
    for scenario in scenarios:
        if scenario == 'loc':
            mean_filter_results = pd.concat([results.filter(like=f'loc_{experiment_idx}').mean(axis=1) for experiment_idx in range(n_splits)],axis=1).mean(axis=1).round(3)
            std_filter_results= pd.concat([results.filter(like=f'loc_{experiment_idx}').std(axis=1,ddof=0) for experiment_idx in range(n_splits)],axis=1).mean(axis=1).round(3)
        else:
            mean_filter_results=results.filter(like=scenario).mean(axis=1).round(3)
            std_filter_results=results.filter(like=scenario).std(axis=1,ddof=0).round(3)
        result_values = mean_filter_results.astype(str) + "±" + std_filter_results.astype(str)
        average_results.loc[f'synth{scenario}',:]=result_values.values
    return average_results
