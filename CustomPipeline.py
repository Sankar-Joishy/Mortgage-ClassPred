# CustomPipeline.py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

class CustomPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, clf, reg):
        self.clf = clf
        self.reg = reg

    def fit(self, X, y_class, y_reg):
        self.clf.fit(X, y_class)
        self.reg.fit(X, y_reg)
        return self

    def predict(self, X):
        y_class_pred = self.clf.predict(X)
        y_reg_pred = self.reg.predict(X)
        return y_class_pred, y_reg_pred
