import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def plot_prediction_density(
        y_true, scores, figsize=(8,5),
        title='Prediction Density Plot',
        colors=['red', 'blue']):

    class_set = sorted(set(y_true))

    x_grid = np.linspace(0, 1, 1000)

    fig, ax = plt.subplots(figsize=figsize)
    for i, value in enumerate(class_set):
        arr = scores[y_true == value]
        kernel = gaussian_kde(arr, bw_method='scott')
        kde = kernel.evaluate(x_grid)
        ax.plot(x_grid, kde, linewidth=2.5, label='Target = {}'.format(value),
                color=colors[i])
        ax.fill_between(x_grid, kde, alpha=0.6, color=colors[i])
    plt.title(title)
    plt.xlabel('Model Score')
    plt.ylabel('Kernel Density')
    plt.legend()
    plt.close(fig)
    return fig