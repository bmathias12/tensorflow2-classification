from binaryclassifier.plotting.colors import THEME_COLORS
from binaryclassifier.plotting import (
    plot_roc_curve, 
    plot_prediction_density, 
    plot_quantiles,
    plot_ks,
    confusion_matrix,
    plot_confusion_matrix,
    plot_precision_recall)


class Analyzer:
    
    def __init__(self, y_true, scores, preds):
        self.y_true = y_true
        self.scores = scores
        self.preds = preds

    def plot_roc_curve(self, color=THEME_COLORS[0], **kwargs):
        return plot_roc_curve(self.y_true, self.scores, color=color, **kwargs)

    def plot_prediction_density(self, colors=THEME_COLORS, **kwargs):
        return plot_prediction_density(self.y_true, self.scores, colors=colors,
            **kwargs)

    def plot_quantiles(self, color=THEME_COLORS[0], **kwargs):
        return plot_quantiles(self.y_true, self.scores, color=color, **kwargs)

    def plot_ks(self, colors=THEME_COLORS, **kwargs):
        return plot_ks(self.y_true, self.scores, colors=colors, **kwargs)

    def plot_confusion_matrix(self, cmap='winter', normalize=False, **kwargs):
        cm = confusion_matrix(self.y_true, self.preds, normalize=normalize)
        return plot_confusion_matrix(cm, cmap=cmap, **kwargs)

    def plot_precision_recall(self, color=THEME_COLORS[0], **kwargs):
        return plot_precision_recall(
            self.y_true, self.scores, color=color, **kwargs)



