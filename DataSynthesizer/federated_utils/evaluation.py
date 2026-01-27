import matplotlib.pyplot as plt
import plotly 
from dython.nominal import associations
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder, LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, precision_score, recall_score, roc_auc_score,r2_score
import pandas as pd
import numpy as np
import os
import networkx as nx

from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import SingleTableMetadata

def SDVMetrics(original_data, synthetic_data, path_metrics=None, path_visualizations=None, learning_scenario='',epsilon=0):
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=original_data)
    
    quality_report = evaluate_quality(
                        real_data=original_data,
                        synthetic_data=synthetic_data,
                        metadata=metadata)
    
    results=quality_report.get_properties()
    
    properties =  results['Property'].values
    
    if path_visualizations is not None:  
        for property in properties:
            name = property.lower().replace(' ', '_')
            fig = quality_report.get_visualization(property_name=property)
            plotly.offline.plot(fig,filename=os.path.join(path_visualizations,f'{name}.html'))
    
    new_entry = pd.DataFrame([{'Property':'Overall Score', 'Score':round(quality_report.get_score(),5)}])
    results= pd.concat([results,new_entry], ignore_index=True)
    results=results.rename(columns={"Score": f"Score {learning_scenario}"})
    
    if path_metrics is not None:    
        results.to_csv(f'{path_metrics}/sdv_metrics_epsilon_{str(epsilon)}.csv',index=False)
    
    return results
    

def HistogramAndHeatmapComparison(train_data, synthetic_data, path_results):
    """
    Function to plot distribution parameters of original training data against synthetic data and also
    heatmap of correlations for training data and synthetic data.

    Args:
        train_data: file of the original training dataset
        synthetic_data: file of the synthetic dataset
        path_results: path to save results
    """

    input_df = pd.read_csv(train_data, skipinitialspace=True)
    syn_df = pd.read_csv(synthetic_data)

    # Label Encoding
    header = list(input_df.columns.values)

    numeric_attributes = syn_df.select_dtypes(include=np.number).columns

    le = preprocessing.LabelEncoder()

    for i in range(len(header)):

        if header[i] not in numeric_attributes:
            input_df.iloc[:, i] = le.fit_transform(list(input_df.iloc[:, i])).astype(np.float64)
            syn_df.iloc[:, i] = le.fit_transform(list(syn_df.iloc[:, i])).astype(np.float64)

    # Check if path for saving results exists, if not creates the corresponding folders
    if not os.path.exists(path_results+'/Plots'):
        os.makedirs(path_results + '/Plots/Histograms')

    correlations_syn = syn_df.corr()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.matshow(correlations_syn, cmap='seismic', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(header), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(header,rotation = 90)
    ax.set_yticklabels(header)
    plt.tight_layout()
    plt.savefig(path_results + '/Plots/heatsyn.png')
    plt.close()

    correlations_real = input_df.corr()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.matshow(correlations_real, cmap='seismic', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(header), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(header,rotation = 90)
    ax.set_yticklabels(header)
    plt.tight_layout()
    plt.savefig(path_results + '/Plots/heatreal.png')
    plt.close()

    for i in range(len(header)):
        plt.hist(input_df.iloc[:, i], color='green', label='Real: ' + header[i])
        plt.legend(fontsize='x-small')
        plt.hist(syn_df.iloc[:, i], histtype='step', color='blue', label='Syn: ' + header[i])
        plt.legend(fontsize='x-small')
        plt.savefig(path_results + '/Plots/Histograms/' + header[i] + '.png')
        plt.close()
        
def TaskUtility(df_real_train, df_syn_train, df_test, label_predictor, path_results = None, type_task = 'Classification', learning_scenario = ''):
    

    # Label Encoding
    header = list(df_real_train.columns.values)
    
    numeric_attributes = df_syn_train.select_dtypes(include=np.number).columns

    le = preprocessing.LabelEncoder()

    for i in range(len(header)):

        if header[i] not in numeric_attributes:
            df_real_train.iloc[:, i] = le.fit_transform(list(df_real_train.iloc[:, i])).astype(np.float64)
            df_test.iloc[:, i] = le.transform(list(df_test.iloc[:, i])).astype(np.float64)
            df_syn_train.iloc[:, i] = le.fit_transform(list(df_syn_train.iloc[:, i])).astype(np.float64)
            
      
    X_train_real= df_real_train.drop(columns=[label_predictor])  
    y_train_real = df_real_train[label_predictor].astype('int')
    
    X_train_syn= df_syn_train.drop(columns=[label_predictor])  
    y_train_syn = df_syn_train[label_predictor].astype('int')
    
    X_test= df_test.drop(columns=[label_predictor])  
    y_test = df_test[label_predictor].astype('int')
          
    if type_task == "Classification":
        rf_real = RandomForestClassifier()
        rf_real.fit(X_train_real,y_train_real)
        
        nb_real = GaussianNB()
        nb_real.fit(X_train_real,y_train_real)
        
        sv_real = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        sv_real.fit(X_train_real,y_train_real)
        
        knn_real = make_pipeline(StandardScaler(), KNeighborsClassifier())
        knn_real.fit(X_train_real,y_train_real)
        
        lr_real = LogisticRegression()
        lr_real.fit(X_train_real,y_train_real)
        
        rf_syn = RandomForestClassifier()
        rf_syn.fit(X_train_syn,y_train_syn)
        
        nb_syn = GaussianNB()
        nb_syn.fit(X_train_syn,y_train_syn)
        
        sv_syn = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        sv_syn.fit(X_train_syn,y_train_syn)
        
        knn_syn =  make_pipeline(StandardScaler(), KNeighborsClassifier())
        knn_syn.fit(X_train_syn,y_train_syn)
        
        lr_syn = LogisticRegression()
        lr_syn.fit(X_train_syn,y_train_syn)
                
        prediction_real_RF = rf_real.predict(X_test)
        prediction_syn_RF  = rf_syn.predict(X_test)
        
        prediction_real_NB = nb_real.predict(X_test)
        prediction_syn_NB = nb_syn.predict(X_test)
        
        prediction_real_SV = sv_real.predict(X_test)
        prediction_syn_SV  = sv_syn.predict(X_test)
        
        prediction_real_KNN = knn_real.predict(X_test)
        prediction_syn_KNN = knn_syn.predict(X_test)
        
        prediction_real_LR = lr_real.predict(X_test)
        prediction_syn_LR = lr_syn.predict(X_test)
        
        algorithms = ['RF','NB', 'SV', 'KNN', 'LR']
        index = ['Original',f'Synthetic {learning_scenario}']
        
        results = pd.DataFrame(columns=algorithms, index=index)
        
        for alg in algorithms:
            real_acc = accuracy_score(y_test,eval(f'prediction_real_{alg}'))
            syn_acc = accuracy_score(y_test,eval(f'prediction_syn_{alg}'))
            results.at['Original',alg] = real_acc
            results.at[f'Synthetic {learning_scenario}',alg] = syn_acc
        
        
    else:
        rf_real = RandomForestRegressor()
        rf_real.fit(X_train_real,y_train_real)
        
        rf_syn = RandomForestRegressor()
        rf_syn.fit(X_train_syn,y_train_syn)
    
    if path_results is not None:
        # Check if path for saving results exists, if not creates the corresponding folders
        if not os.path.exists(path_results):
            os.makedirs(path_results)
        results.to_csv(f'{path_results}/task_utility.csv',index=True)    
    
    return results


def plot_bayesian_network(bayesian_network):
    graph = nx.DiGraph()
    edges_dag = []
    for child, parents in bayesian_network:
        for parent in parents:
            edges_dag.append((parent,child))
    graph.add_edges_from(edges_dag)
    for layer, nodes in enumerate(nx.topological_generations(graph)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            graph.nodes[node]["layer"] = layer
    pos = nx.multipartite_layout(graph, subset_key="layer")
    pos_higher = {}
    y_off = -0.04  # offset on the y axis
    for k, v in pos.items():
        pos_higher[k] = (v[0], v[1]+y_off)
    nx.draw(graph, pos, arrows=True, style='dashed', width=0.5)
    nx.draw_networkx_labels(graph, pos_higher, font_size=8)
    plt.tight_layout()
    return plt    
    
