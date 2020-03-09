import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def confusion_matrix(y_true, preds, normalize=False):
    y_true = pd.Series(y_true, name='Actual')
    preds = pd.Series(preds, name='Predictions')
    return pd.crosstab(y_true, preds, normalize=normalize)

def plot_confusion_matrix(
        df, title='Confusion Matrix', cmap='winter', 
        figsize=(6,6), **kwargs):
    
    fig, ax = plt.subplots(figsize=figsize)

    # Check if confusion matrix is counts or proportions
    cm_is_int = np.all(df.dtypes == 'int')

    sns.heatmap(df, cmap=cmap, annot=True,
                fmt='d' if cm_is_int else '0.3f', cbar=False, linewidths=0.3,
                annot_kws={'fontsize':16, 'fontweight':'bold'}, **kwargs)
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)
    plt.title(title, fontsize=16)
    plt.close(fig)
    return fig