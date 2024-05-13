import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from pandas.core.common import random_state
import numpy as np

# load catboost_df
catboost_df = pd.read_csv('datasets/catboost_df.csv', index_col=0)
# drop label name_x and name_y
catboost_df = catboost_df.drop(['name_x', 'name_y'], axis=1)
# get the categorical and float features
cat_features = list(catboost_df.select_dtypes(include=['object']).columns)
float_features = list(catboost_df.select_dtypes(include=['float64']).columns)

for feature in float_features:
    # Fill NaN values with the mean of non-missing values in the same column
    mean_value = catboost_df[feature].mean()
    catboost_df[feature].fillna(mean_value, inplace=True)

for feature in cat_features:
    catboost_df[feature] = catboost_df[feature].astype(str)

# create test and train set
X, y = catboost_df.drop('interaction', axis=1), catboost_df['interaction']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.6, random_state=42)

inference = CatBoostClassifier()
inference.load_model("models/catboost_model2.cbm")

y_pred = inference.predict_proba(X_test)
y_pred = y_pred[:, 1]
y_pred_binary = np.where(y_pred > 0.5, 1, 0)
print(f"Test AUC_ROC score = {roc_auc_score(y_test, y_pred)}")
print(f"Accuracy Score= {accuracy_score(y_test, y_pred_binary)}")
