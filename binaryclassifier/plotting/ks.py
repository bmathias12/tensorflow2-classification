import matplotlib.pyplot as plt

from binaryclassifier.statistics.quantiles import (
    get_bins, get_quantiles_from_bins)
from binaryclassifier.statistics.ks import ks_table


def plot_ks(
    y_true, scores, q=10, figsize=(8,5), title='K-S Plot',
    colors=['red', 'blue']):

    if q==10:
        quant_type = 'Decile'
    elif q==20:
        quant_type = 'Ventile'
    else:
        quant_type = 'Quantile'

    # Get quantiles from probabilities
    bins = get_bins(scores, q=q, adjust_endpoints=True)
    quantiles = get_quantiles_from_bins(scores, bins, one_high=True)

    # Create K-S table
    ks_df = ks_table(y_true, quantiles)
    idx = sorted(set(quantiles))

    pos_pct = ks_df['pos_pct_cum']
    neg_pct = ks_df['neg_pct_cum']

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(idx, pos_pct, linewidth=2.5, label='Positive Examples', color=colors[0])
    ax.plot(idx, neg_pct, linewidth=2.5, label='Negative Examples', color=colors[1])

    # Add K-S distance segment to plot
    ks_quantile = ks_df['ks_idx'].idxmax()
    ks_dist_points = ks_df.iloc[ks_quantile - 1, 3:5]
    ax.vlines(x=ks_quantile,
          ymin=ks_dist_points.min(),
          ymax=ks_dist_points.max(),
          color='black',
          linestyle='--',
          linewidth=2,
          label='K-S = {:0.3f}'.format(ks_df['ks_idx'].max()))

    ax.set_xticks(idx)
    plt.title(title)
    plt.xlabel(quant_type)
    plt.ylabel('Cumulative Percent')
    plt.legend()
    plt.close(fig)
    return fig