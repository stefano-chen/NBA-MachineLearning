import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

from data_processing import data_split


def model(estimator, data: pd.DataFrame, params: dict):
    x_train, x_test, y_train, y_test = data_split(data, 'HOME_TEAM_WINS')
    classifier = GridSearchCV(estimator, param_grid=params, scoring='accuracy', cv=5, n_jobs=-1)
    classifier.fit(x_train, y_train)
    training_result = classifier.cv_results_
    x_label = [item for item in training_result['params']]
    y_values = training_result['mean_test_score']
    print(f'{classifier.best_estimator_} Ready')
    y_pred = classifier.predict(x_test)
    y_pred_prob = None
    if hasattr(classifier, 'predict_proba') and callable(classifier.predict_proba):
        y_pred_prob = classifier.predict_proba(x_test)[:, 1]
    else:
        y_pred_prob = classifier.decision_function(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cmatrix = confusion_matrix(y_test, y_pred)
    fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    return {
        'name': estimator.__class__.__name__,
        'training': classifier.best_score_,
        'accuracy': accuracy,
        'confusion': cmatrix,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc
    }
