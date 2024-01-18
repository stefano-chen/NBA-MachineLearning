import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

n_colors = 6

colors = []

while n_colors > 0:
    color = (round(random.uniform(0, 1), 1), round(random.uniform(0, 1), 1), round(random.uniform(0, 1), 1))
    if color not in colors:
        colors.append(color)
        n_colors -= 1


def corr_heatmap(data: pd.DataFrame):
    plt.subplots(figsize=(18, 9))
    plt.subplots_adjust(left=0.14, bottom=0.22, right=1, top=0.95)
    corr = data.corr()
    sns.heatmap(corr, cmap='Blues', annot=True, center=True, linewidths=0.5, vmin=-1, vmax=1, fmt='.2f')
    plt.show()


def confusion_matrix_display(matrix, names):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 9))  # type: plt.Figure, plt.Axes
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.96, top=0.92, wspace=0.25, hspace=0.25)
    for i, (name, ax) in enumerate(zip(names, axes.flatten())):
        ax.set_title(name)
        ConfusionMatrixDisplay(matrix[i], display_labels=np.array(['HOME_LOSS', 'HOME_WIN'])).plot(ax=ax,
                                                                                                   colorbar=False)
    plt.show()


def accuracy_plot(names, accuracies: list, title: str):
    fig, ax = plt.subplots(figsize=(10, 9))  # type: plt.Figure, plt.Axes
    bars = ax.bar(x=names, height=accuracies, color=colors)
    ax.bar_label(bars)
    ax.set_title(title)
    plt.show()


def roc_auc_plot(names, fpr, tpr, auc):
    fig, ax = plt.subplots(figsize=(10, 9))  # type: plt.Figure, plt.Axes
    for i, name in enumerate(names):
        RocCurveDisplay(estimator_name=name, fpr=fpr[i], tpr=tpr[i], roc_auc=auc[i]).plot(ax, color=colors[i])
    plt.show()
