from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

from data_processing import *
from data_visualizzation import *
from models import *


def compare_techniques(data: pd.DataFrame):
    results = []
    results.append(model(DecisionTreeClassifier(), data, {
        'criterion': ['gini', 'entropy', 'log_loss']
    }))
    results.append(model(RandomForestClassifier(), data, {
        'n_estimators': [50, 100, 150, 200, 250]
    }))
    results.append(model(LinearSVC(dual='auto'), data, {
        'C': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    }))
    results.append(model(GaussianNB(), data, {}))
    results.append(model(KNeighborsClassifier(), data, {
        'n_neighbors': [50, 100, 150, 200, 250]
    }))
    results.append(model(DummyClassifier(), data, {}))
    names = [result['name'] for result in results]
    accuracies = [round(result['accuracy'], 4) for result in results]
    fpr = [result['fpr'] for result in results]
    tpr = [result['tpr'] for result in results]
    auc = [result['auc'] for result in results]
    confusions = [result['confusion'] for result in results]
    accuracy_plot(names, accuracies)
    roc_auc_plot(names, fpr, tpr, auc)
    confusion_matrix_display(confusions, names)


if __name__ == '__main__':
    dataframe = load_data('./datasets/nba.csv')
    corr_heatmap(dataframe)
    compare_techniques(dataframe)
