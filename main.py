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
        'n_estimators': [200, 250, 300, 350, 400]
    }))
    results.append(model(LinearSVC(dual='auto'), data, {
        'C': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    }))
    results.append(model(GaussianNB(), data, {}))
    results.append(model(KNeighborsClassifier(), data, {
        'n_neighbors': [50, 100, 150, 200, 250]
    }))
    results.append(model(DummyClassifier(), data, {}))
    for result in results:
        print(result)


if __name__ == '__main__':
    dataframe = load_data('./datasets/nba.csv')
    corr_heatmap(dataframe)
    compare_techniques(dataframe)
