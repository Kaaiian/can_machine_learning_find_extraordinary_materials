import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import f1_score


class Model():
    def __init__(self, estimator, param_candidates, classification=False):
        self.estimator = estimator
        self.param_candidates = param_candidates
        self.classification = classification

    def get_grid_params(self, X, y):
        cv = KFold(n_splits=5, shuffle=True, random_state=1)
        estimator = self.estimator()
        grid = GridSearchCV(estimator=estimator,
                            param_grid=self.param_candidates,
                            cv=cv)
        n = 1000
        if self.classification:
            n = 3000
        if X.shape[0] > n:
            print('reducing dataset to {:0.0f} for grid search'.format(n))
            X = X.sample(n, random_state=1)
            y = y[X.index]
        grid.fit(X, y)
        self.best_params = grid.best_params_

    def fit(self, X, y):
        print('starting grid search for:', self.estimator())
        self.get_grid_params(X, y)
        if self.classification:
            try:
                self.model = self.estimator(probability=True,
                                            **self.best_params)
            except TypeError:
                self.model = self.estimator(**self.best_params)
        else:
            self.model = self.estimator(**self.best_params)
        self.model.fit(X, y)

    def predict(self, X):
        if self.classification:
            y_pred = self.model.predict_proba(X)
            y_pred = pd.Series([probability[1] for probability in y_pred])
        else:
            y_pred = self.model.predict(X)
        return y_pred

    def predict_proba(self, X):
        if self.classification:
            y_pred = self.model.predict_proba(X)
            y_pred = pd.Series([probability[1] for probability in y_pred])
        else:
            y_pred = self.model.predict(X)
            y_pred = y_pred - y_pred.mean()
            y_pred = y_pred / y_pred.std()
            y_pred = 1/(1+np.exp(-y_pred))
        return y_pred

    def optimize_threshold(self, y_train_labeled, y_train_pred):
        """Given a DataFrame of labels and predictions, return the
         optimal threshold for a high F1 score"""
        y_train_ = y_train_labeled.copy()
        y_train_pred_ = pd.Series(y_train_pred).copy()
        opt_thresh = 0.5
        y_train_pred_[y_train_pred < opt_thresh] = 0
        y_train_pred_[y_train_pred >= opt_thresh] = 1
        f1score_max = f1_score(y_train_, y_train_pred_)
        for threshold in np.arange(0.1, 1, 0.1):
            diff = (max(y_train_pred) - min(y_train_pred))
            threshold = min(y_train_pred) + threshold * diff
            y_train_pred_[y_train_pred < threshold] = 0
            y_train_pred_[y_train_pred >= threshold] = 1
            f1score = f1_score(y_train_, y_train_pred_)
            if f1score > f1score_max:
                f1score_max = f1score
                opt_thresh = threshold
        self.threshold = opt_thresh


#estimator = KNeighborsRegressor()
#svr_grid_params = {'C': np.logspace(2, 4, 5),
#                   'gamma': np.logspace(-3, 1, 5)}
#grid = GridSearchCV(estimator=estimator,
#                    param_grid=svr_grid_params)
#grid.fit(np.array([0, 1, 23]).reshape(-1, 1), np.array([0, 1, 23]).reshape(-1, 1))
