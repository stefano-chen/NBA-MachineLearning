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
    x_train, x_test, y_train, y_test = data_split(data, 'HOME_TEAM_WINS')
    results.append(model(DecisionTreeClassifier(), x_train, x_test, y_train, y_test, {
        'criterion': ['gini', 'entropy', 'log_loss']
    }, save_to_file=True, load_from_file=True))
    results.append(model(RandomForestClassifier(), x_train, x_test, y_train, y_test, {
        'n_estimators': [50, 100, 150, 200, 250]
    }, save_to_file=True, load_from_file=True))
    results.append(model(LinearSVC(dual='auto'), x_train, x_test, y_train, y_test, {
        'C': [1e-5, 1e-4, 1e-3, 1e-1, 1]
    }, save_to_file=True, load_from_file=True))
    results.append(model(GaussianNB(), x_train, x_test, y_train, y_test, {}, save_to_file=True, load_from_file=True))
    results.append(model(KNeighborsClassifier(), x_train, x_test, y_train, y_test, {
        'n_neighbors': [50, 100, 150, 200, 250]
    }, save_to_file=True, load_from_file=True))
    results.append(model(DummyClassifier(), x_train, x_test, y_train, y_test, {}, save_to_file=True, load_from_file=True))
    names = [result['name'] for result in results]
    training_scores = [result['training'] for result in results]
    accuracies = [round(result['accuracy'], 4) for result in results]
    fpr = [result['fpr'] for result in results]
    tpr = [result['tpr'] for result in results]
    auc = [result['auc'] for result in results]
    confusions = [result['confusion'] for result in results]
    accuracy_plot(names, training_scores, 'Training')
    accuracy_plot(names, accuracies, 'Testing')
    roc_auc_plot(names, fpr, tpr, auc)
    confusion_matrix_display(confusions, names)


if __name__ == '__main__':
    dataframe = load_data('./datasets/nba.csv')
    corr_heatmap(dataframe)
    compare_techniques(dataframe)
