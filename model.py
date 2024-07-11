import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


from sklearn.base import BaseEstimator, TransformerMixin
import pickle

data = pd.read_csv('converted.csv')

# Split data into features and target
X = data.drop(['EverDelinquent', 'Prepayment'], axis=1)
y_class = data['EverDelinquent']
y_reg = data['Prepayment']

X.replace([np.inf, -np.inf], np.nan, inplace=True)


threshold = 1e10
X = X.applymap(lambda x: np.nan if abs(x) > threshold else x)

X.dropna(inplace=True)
y_class = y_class[X.index]
y_reg = y_reg[X.index]


# Split into training and testing sets
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)


from sklearn.ensemble import RandomForestClassifier

# Initialize and train the classification model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_class_train)

# Make predictions
y_class_pred = clf.predict(X_test)


# Filter training data based on classification predictions
X_train_reg = X_train[y_class_train == 1]
y_reg_train_filtered = y_reg_train[y_class_train == 1]

# Filter test data based on classification predictions
X_test_reg = X_test[y_class_pred == 1]
y_reg_test_filtered = y_reg_test[y_class_pred == 1]

# Initialize and train the regression model
reg = LinearRegression()
reg.fit(X_train_reg, y_reg_train_filtered)

# Make predictions
y_reg_pred = reg.predict(X_test_reg)

class CustomPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, clf, reg):
        self.clf = clf
        self.reg = reg

    def fit(self, X, y_class, y_reg):
        self.clf.fit(X, y_class)
        X_filtered = X[y_class == 1]
        y_reg_filtered = y_reg[y_class == 1]
        self.reg.fit(X_filtered, y_reg_filtered)
        return self

    def predict(self, X):
        y_class_pred = self.clf.predict(X)
        X_filtered = X[y_class_pred == 1]
        y_reg_pred = self.reg.predict(X_filtered)
        return y_class_pred, y_reg_pred

# Create and fit the custom pipeline
pipeline = CustomPipeline(clf=RandomForestClassifier(random_state=42), reg=LinearRegression())
pipeline.fit(X_train, y_class_train, y_reg_train)

# Make predictions
y_class_pred, y_reg_pred = pipeline.predict(X_test)

# Save the pipeline
with open('pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print('Pipeline has been pickled and saved to pipeline.pkl')

# Load and use the pipeline for predictions
#loaded_pipeline = pickle.load('combined_pipeline.pkl')
#y_class_pred, y_reg_pred = loaded_pipeline.predict(X_test)
