import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold

import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")


def preprocess(X):
    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="mean")), ("scale", MinMaxScaler())]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one-hot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    numerical_features = X.select_dtypes(include="number").columns
    categorical_features = X.select_dtypes(include="object").columns

    # X_train, X_valid, y_train, y_valid = train_test_split(
    #    X, y, test_size=.3, random_state=1121218)

    numeric_pipeline.fit_transform(X.select_dtypes(include="number"))
    categorical_pipeline.fit_transform(X.select_dtypes(include="object"))

    prep = ColumnTransformer(
        transformers=[
            ("number", numeric_pipeline, numerical_features),
            ("category", categorical_pipeline, categorical_features),
        ]
    )

    X = prep.fit_transform(X)

    return X


loan_data = pd.read_csv("Loan-Prediction/data/train.csv")

X = loan_data.drop(["Loan_Status", "Loan_ID"], axis=1)
y = loan_data.Loan_Status

X = preprocess(X)
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=1121218
)

xgb_cl = xgb.XGBClassifier()

xgb_cl.fit(X_train, y_train)
preds = random_search.predict(X_test)

print(accuracy_score(y_test, preds))

##############################################################
params = {
    "min_child_weight": [1, 5, 10],
    "gamma": [0.5, 1, 1.5, 2, 5],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.02, 0.005],
    "n_estimators": [100, 200, 400, 600],
}

xgb_ = xgb.XGBClassifier()


random_search = GridSearchCV(
    xgb_,
    params,
    verbose=3,
    n_jobs=-1
)

random_search.fit(X, y)
#############################################################







def generate_result(model):
    test = pd.read_csv("Loan-Prediction/data/test.csv")
    test = test.drop("Loan_ID", axis=1)

    test = preprocess(test)

    res = model.predict(test)

    result = pd.read_csv("Loan-Prediction/data/ss.csv")
    result["Loan_Status"] = res

    rm = {1: "Y", 0: "N"}
    result = result.replace({"Loan_Status": rm})

    result.to_csv("Pred_v2.csv", index=False)


from sklearn.metrics import classification_report, confusion_matrix 

print(classification_report(y_test, preds)) 

print(confusion_matrix(y_test, preds)) 