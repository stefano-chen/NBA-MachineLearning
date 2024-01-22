import time

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from joblib import dump, load


def model(estimator, x_train, x_test, y_train, y_test, params: dict, load_from_file=False, save_to_file=False):
    classifier = None
    if not load_from_file:
        classifier = GridSearchCV(estimator, param_grid=params, scoring='accuracy', cv=5, n_jobs=-1)
        start_time = time.time()
        classifier.fit(x_train, y_train)
        end_time = time.time()
        print(f'{classifier.best_estimator_} Ready [{end_time - start_time}s]')
    else:
        classifier = load(f'./classifiers/{estimator.__class__.__name__}.joblib')
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
    if save_to_file:
        dump(classifier, f'./classifiers/{estimator.__class__.__name__}.joblib')
    return {
        'name': estimator.__class__.__name__,
        'training': classifier.best_score_,
        'accuracy': accuracy,
        'confusion': cmatrix,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc
    }
