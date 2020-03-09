import matplotlib.pyplot as plt

from binaryclassifier.statistics.quantiles import (
    get_bins, get_quantiles_from_bins)


def plot_quantiles(
        y_true, scores, q=10, figsize=(8,5),
        title='Quantile Bar Graph', color=None):

    if q==10:
        quant_type = 'Decile'
    elif q==20:
        quant_type = 'Ventile'
    else:
        quant_type = 'Quantile'

    # Get quantiles from probabilities
    bins = get_bins(scores, q=q, adjust_endpoints=True)
    quantiles = get_quantiles_from_bins(scores, bins, one_high=True)

    # Calculate event rate for each quantile
    quantile_labels = sorted(set(quantiles))
    rates = []
    for q in quantile_labels:
        arr = y_true[quantiles == q]
        pos_vals = arr.sum()
        rate = pos_vals / float(len(arr))
        rates.append(rate)

    # Plot event rate for each quantiles
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(quantile_labels, rates, align='center', color=color)
    ax.set_xticks(quantile_labels)
    plt.title(title)
    plt.ylabel('Event Rate')
    plt.xlabel(quant_type)
    plt.close(fig)
    return fig