
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, RandomizedSearchCV
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from pandas.core.common import random_state

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
    X, y, test_size=0.2, random_state=42)

catb_model = CatBoostClassifier(random_state=42, task_type="GPU", max_ctr_complexity=1, boosting_type="Plain",
                                cat_features=cat_features, gpu_ram_part=0.4)
catb_param = {
    'max_depth': [6],
    'learning_rate': [0.01],
    'reg_lambda': [2.5],
    'n_estimators': [1000],
}


# pool_train = Pool(X_train, y_train, cat_features = cat_features)
# pool_test = Pool(X_test, cat_features = cat_features)

# grid search
grid_search = HalvingGridSearchCV(
    catb_model, catb_param, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Done")

best_model = grid_search.best_estimator_
best_model.save_model('models/catboost_model2.cbm')


# print best parameters
print(grid_search.best_params_)
# print best score
print(grid_search.best_score_)


y_p = grid_search.predict_proba(X_test)
print(f"Test AUC_ROC score = {roc_auc_score(y_test, y_p[:, 1])}")

print("---------------------Done--------------------------------")
