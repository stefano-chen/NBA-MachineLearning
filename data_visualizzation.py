import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def corr_heatmap(data: pd.DataFrame):
    plt.subplots(figsize=(18, 9))
    plt.subplots_adjust(left=0.14, bottom=0.22, right=1, top=0.95)
    corr = data.corr()
    sns.heatmap(corr, cmap='Blues', annot=True, center=True, linewidths=0.5, vmin=-1, vmax=1, fmt='.2f')
    plt.show()
