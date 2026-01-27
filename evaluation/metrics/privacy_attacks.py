import itertools
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_absolute_percentage_error, r2_score
from evaluation.utils import average_results, majority_voting
from pandas.api.types import is_numeric_dtype


def attribute_disclosure_as_ml_task(real_data,synth_data, quasi_identifiers, key_length, target, categorical_columns, preprocessor = None, decimals=3):
    """
    Produces results for attribute disclosure attacks based on quasi-identifiers key_length, computes all possible combinations and returns the mean and std accuracy of several classifiers. 
    
    Parameters
    -----------
    real_data : pandas.DataFrame
        Dataframe containing the real data.
    synth_data : pandas.DataFrame
        Dataframe containing the synthetic data.
    quasi_identifiers : list
        List of attributes (possibly) known by the attacker.       
    key_length: int 
        Number of attributes known by the attacker.
    target: str
        Sensitive attribute the attacker tries to infer. 
    categorical_columns: list
        List of categorical attributes in the real and synthetic data.
    preprocessor: (ColumnTransformer,optional)
        ColumnTransformer from sklearn with specific steps to preprocess the data. 
    
    Returns
    -----------
    results: dict
        Dictionary with a dataframe per each metric with mean and std scores from the predictions obtained for the target value given the combinations of quasi-identifiers.
        
    """
    
    key_list = list(itertools.combinations(quasi_identifiers, key_length))
    
    # The original dataset is used as reference (Upper Bound)
    data_evaluated = {
        'Original': real_data,
        'Synthetic':synth_data
    }
    
    # Define estimators used to estimate disclosure
    if target in categorical_columns:
        
        estimators = {
            'RF': RandomForestClassifier(random_state=42),
            'SVM': SVC(),
            'NB':  GaussianNB(),
            'KNN': KNeighborsClassifier(),
            'LR':  LogisticRegression()
        }
        
        # The dummy classifier is used as reference (Lower Bound)
        baseline =  DummyClassifier()
        
        metrics = {
            'acc': accuracy_score,
            'f1': f1_score
        }
        
        ensemble_function = majority_voting
        
    else:
        if is_numeric_dtype(real_data[target]):
            estimators = {
                'LR': LinearRegression(),
                'SVR': SVR(kernel='rbf'),
                'MLP': MLPRegressor(solver='adam')    
            }
            
            # The dummy regressor is used as reference (Lower Bound)
            baseline =  DummyRegressor()
            
            metrics = {
                'mae': mean_absolute_error,
                'r2': r2_score,
                'mape': mean_absolute_percentage_error
            }
            
            ensemble_function = average_results
            
        else:
            print(f'The target variable {target} type is not recognized. Expected types: categorical or numeric.')
            raise Exception(f'The target variable {target} type is not recognized. Expected types: categorical or numeric.')
        
    
    # Preprocessing step in case is not defined. 
    if preprocessor == None:
        
        numerical_columns= [ col for col in real_data.columns if (col not in categorical_columns and col != target)]
        
        # Preprocess numerical columns
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Preprocess categorical columns
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        # Build transformer with preprocessing pipelines.
        preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", num_transformer, numerical_columns),
                ("categorical", cat_transformer, categorical_columns),
            ],
            verbose_feature_names_out = False,
            remainder='passthrough'
        )   
    
    results = {metric_name: pd.DataFrame() for metric_name in metrics.keys()}
    
            
    for data_name,dataset in data_evaluated.items():
        
        print('started')
        train_data=pd.DataFrame(preprocessor.fit_transform(dataset))
        train_data.columns= preprocessor.get_feature_names_out() 
        test_data=pd.DataFrame(preprocessor.transform(real_data),columns=preprocessor.get_feature_names_out()) 
        
        # Keys combinations    
        for i,key in enumerate(key_list):
    
            X_train = train_data[list(key)]
            X_test = test_data[list(key)]
            
            y_train = train_data.loc[:,target].values          
            y_test = test_data.loc[:,target].values
            
            if len(np.unique(y_test)) == 2:
                average='binary'
            else:
                print('macro')
                average='macro'
            
            # Classifiers training 
            y_pred_all = []
            for model_name,model in estimators.items():
                model.fit(X_train,y_train)
                y_pred=model.predict(X_test)
                y_pred_all.append(y_pred)
                
            # Models
            all_models = list(estimators.keys())
            
            # Ensemble results
            all_models.append('ENS')
            y_pred_all.append(ensemble_function(y_pred_all))
            
            for model_name,y_pred  in zip(all_models,y_pred_all):    
                for metric in metrics.keys():
                    metric_function = metrics[metric]
                    if metric == 'f1':
                        score = metric_function(y_test,y_pred,average=average)
                    else:
                        score = metric_function(y_test,y_pred)  
                    results[metric].at[f'{data_name}_{metric}',f'key{i}-{model_name}'] = score
                 
        # Baseline
        baseline.fit(X_train,y_train)
        y_pred=baseline.predict(X_test)
        for metric in metrics.keys():
            metric_function = metrics[metric]
            if metric == 'f1':
                score = metric_function(y_test,y_pred,average=average)
            else:
                score = metric_function(y_test,y_pred)  
            results[metric].at[f'{data_name}_{metric}','Dummy'] = round(score,decimals)
    
        
    # Average results
    for model_name in all_models:
        for metric in metrics.keys():
            results[metric][f'{model_name}_mean']=results[metric].filter(regex=model_name).mean(axis=1).round(decimals)
            results[metric][f'{model_name}_std']=results[metric].filter(regex=model_name).std(axis=1).round(decimals)
            
    for metric in metrics.keys():  
        results[metric]=results[metric][results[metric].columns.drop(list(results[metric].filter(regex='key')))]
        results[metric][f'ROW_mean']=results[metric].filter(regex='mean').mean(axis=1).round(decimals)
        results[metric][f'ROW_std']=results[metric].filter(regex='std').std(axis=1).round(decimals)
            
    return results
            
def compute_mean_attribute_disclosure(attr_disc_results: Dict, save_dir: str = None):
    '''
    Compute mean attribute disclosure over several runs. 
    
    Parameters
    ----------
    attr_disc_results: Dict 
        Dictionary containing results from attribute disclosure experiments from several runs and several scenarios.
    save_dir: str (optional)
        Path to save the results
        
    Returns
    -------
    result_dict: List
        List containing dictionaries with mean results from multiple attribute disclosure experiments.    
    
    '''
    all_results = []
    for scenario in attr_disc_results.keys():
        results_scenario = attr_disc_results[scenario]
        if len(results_scenario)>0:
            result_dict={}
            for key in results_scenario[0].keys():
                result_dict[key] = pd.concat([ dictionary[key] for dictionary in results_scenario]).groupby(level=0).mean()
                if save_dir != None:
                    result_dict[key].to_csv(f'{save_dir}/attribute_disclosure_{key}_scenario{scenario}.csv')
            all_results.append(result_dict)
    return all_results

    
    