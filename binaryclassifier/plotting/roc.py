import matplotlib.pyplot as plt
import sklearn.metrics as metrics

def plot_roc_curve(y_true, scores, figsize=(8,5), title='ROC Curve', color=None):
    fpr, tpr, _ = metrics.roc_curve(y_true, scores, pos_label=1)
    auc = metrics.roc_auc_score(y_true, scores)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, label='Area Under Curve: {:0.3f}'.format(auc),
            color=color, linewidth=2)
    ax.plot([0,1], [0,1], color='black', linestyle='--')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.close(fig)
    return fig