
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, accuracy_score, r2_score,f1_score, mean_absolute_error, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVR 


def machine_learning_evaluation(df_train, 
                                df_test, 
                                target, 
                                target_labels,
                                metrics, 
                                dataset_name,
                                categorical_features=None, 
                                task='Classification'):
    
    X_train = df_train.drop(target, axis=1)
    y_train = df_train[target]
    
    X_test = df_test.drop(target, axis=1)
    y_test = df_test[target]
    
    # Select column types
    numerical_features = X_train.select_dtypes(include=np.number)
    if categorical_features is None:
        categorical_features = X_train.select_dtypes(include=['object'])
        
    # Create transformers for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # Create LabelEncoder with explicit classes
    label_encoder = LabelEncoder()
    label_encoder.classes_ = target_labels

             
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features.columns),
            ('cat', categorical_transformer, categorical_features.columns)
    ])

    # Perform preprocessing
    X_train=preprocessor.fit_transform(X_train)
    X_test=preprocessor.transform(X_test)
    #y_train=label_encoder.transform(y_train)
    #y_test=label_encoder.transform(y_test)

    if task == 'Classification':
        models = {
                    'rf': RandomForestClassifier(random_state=42), 
                    'nb': GaussianNB(),
                    'knn': KNeighborsClassifier(),
                    'dummy': DummyClassifier(strategy="most_frequent")
                }

        metrics_dict = {
                    'accuracy': accuracy_score,
                    'balanced_accuracy': balanced_accuracy_score,
                    'f1': f1_score,
                    'precision': precision_score,
                    'recall': recall_score,
                    'auc': roc_auc_score
                                 
                }

        # In case, we have a multiclass classification, set average to 'macro'
        args = {'average': 'macro'} if len(np.unique(y_test))> 2 else {'average': 'binary'}
        
    else:
        models = {
                    'rf': RandomForestRegressor(), 
                    'mlp': MLPRegressor(),
                    'svc': SVR(),
                    'knn': KNeighborsRegressor()
                }

        metrics_dict = {
                    'r2': r2_score,
                    'mae': mean_absolute_error           
                }

    # DataFrame to store metric results
    results = pd.DataFrame(index=[dataset_name])

    # Evaluation of all the models
    for name,model in models.items():
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        for metric in metrics:
            if metric in metrics_dict.keys():
                metric_function=metrics_dict[metric]
                if metric in ['balanced_accuracy','accuracy','auc']:
                    results.at[dataset_name,f'{name}-{metric}'] = metric_function(y_test,y_pred)
                else:
                    try:
                        results.at[dataset_name,f'{name}-{metric}'] = metric_function(y_test,y_pred,**args)
                    except Exception as e:
                        print(f'this is the error {e}')
            else:
                raise ValueError(f"Invalid metric name: {metric}")
            
    return results
    
def generate_average_utility_report(df_results, metrics, n_splits, scenarios=['fed','cen','loc']):
    '''Function the converts a dataframe with the metrics of ML models 
    for real and synthetic datasets into average results.
    '''
    
    scenarios_list = ['synth' + sub for sub in scenarios] 
    
    comparison_list=['real']+ scenarios_list
    
    average_results= pd.DataFrame(index=comparison_list,columns=df_results.columns)
    
    for case in comparison_list:
        
        if case == 'loc':
            
            mean_values= pd.concat([df_results[df_results.index.str.contains(f'loc_{experiment_idx}')].mean() for experiment_idx in range(n_splits)],axis=1).mean(axis=1).round(3)
            std_values = pd.concat([df_results[df_results.index.str.contains(f'loc_{experiment_idx}')].std(doff=0) for experiment_idx in range(n_splits)],axis=1).mean(axis=1).round(3)
            
        else:
            df_filtered=df_results[df_results.index.str.contains(case)]
        
            # Calculate mean and standard deviation for each column
            mean_values = df_filtered.mean().round(3)
            std_values = df_filtered.std(ddof=0).round(3)
            
        # Join mean and std of each column into a single string with a "±"
        result_values = mean_values.astype(str) + "±" + std_values.astype(str)
        
        average_results.loc[case]=result_values
        
        for metric in metrics:
            mean_values.drop([f'dummy-{metric}'],axis=0,inplace=True)
            df_all_avg=mean_values.filter(like=metric).mean().round(3)
            df_all_std=mean_values.filter(like=metric).std().round(3)
            average_results.at[case,f'all-{metric}'] = f'{df_all_avg} ± {df_all_std}'
        
    return average_results