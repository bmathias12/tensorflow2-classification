import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve


def plot_precision_recall(
        y_true, scores, figsize=(8,5),
        title='Precision-Recall Curve',
        color=None, mark_threshold=None):
    
    precision, recall, thresholds = precision_recall_curve(
        y_true, scores)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    plt.plot(recall, precision, color=color, linewidth=2)
    
    if mark_threshold:
        mark = np.argmin(np.abs(thresholds - mark_threshold))
        p = precision[:len(precision)-1][mark]
        r = recall[:len(recall)-1][mark]
        threshold_label = f'Score threshold = {mark_threshold}'
        plt.plot(r,p, marker='o', markersize=10, color='black',
                 fillstyle='none', mew=2,
                 label=threshold_label)
        plt.legend()
    
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title('Precision-Recall Curve')
    
    plt.close(fig)
    return fig