import os 
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sdmetrics.visualization import get_column_plot
from evaluation.metrics.privacy_distance import all_distances_closest_records


def plot_univariate_distributions(real_data,synth_data,categorical_columns,save_dir):
    
    if not os.path.exists(f'{save_dir}/plots'):
        os.makedirs(f'{save_dir}/plots')  

    for col in real_data.columns:
        
        if col in categorical_columns:
            plot_type='bar'

        else:
            plot_type='distplot'

        fig = get_column_plot(
            real_data=real_data,
            synthetic_data=synth_data,
            column_name=col,
            plot_type=plot_type,
            
        ).update_layout(
            title='',
            font=dict(
            size=10,  # Set the font size here
            )
        )
    
        fig.write_image(f"{save_dir}/plots/histogram_{col}.pdf")
        
        
def plot_distance_closest_record_distribution(real_data,synthetic_data,holdout_data,categorical_colunms, numerical_columns, save_dir):

    distances_train=all_distances_closest_records(real_data,synthetic_data,categorical_colunms,numerical_columns,k=1)
    distances_test=all_distances_closest_records(holdout_data,synthetic_data,categorical_colunms,numerical_columns,k=1)

    df = pd.concat(axis=0, ignore_index=True, objs=[
       pd.DataFrame.from_dict({'DCR': distances_train.flatten(), 'Comparison': 'Synthetic data vs Training data'}),
       pd.DataFrame.from_dict({'DCR': distances_test.flatten(), 'Comparison': 'Synthetic data vs Holdout data'})
    ])

          
    fig, ax = plt.subplots()
    sns.histplot(
        data=df, x='DCR', hue='Comparison', multiple='dodge',
        bins=40, ax=ax, binrange=(0,6)
    )
    plt.tight_layout()
    plt.savefig(f'{save_dir}/dcr_comparison_real_holdout.png')
        

