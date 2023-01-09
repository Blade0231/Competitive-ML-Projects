#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

# %%
# %%
def prep_data(df, training_cols):
    train = df.copy()

    train.date = pd.to_datetime(train.date)

    train = pd.get_dummies(train, columns=["country"])
    train = pd.get_dummies(train, columns=["store"])
    train = pd.get_dummies(train, columns=["product"])

    train["weekday"] = train["date"].dt.weekday.astype(np.int8)
    train["week_no"] = train["date"].dt.isocalendar().week.astype(np.int8)
    train["month"] = train["date"].dt.month.astype(np.int8)
    train["year"] = train["date"].dt.year.astype(np.int)

    train["Weekend"] = train["weekday"].apply(lambda x: 1 if x > 4 else 0)

    try:
        t_pred = train["num_sold"]
    except:
        t_pred = None
    t_data = train[training_cols]

    # scaler = MinMaxScaler()
    # scaler.fit(t_data)
    # t_data = scaler.transform(t_data)

    return t_data, t_pred


def plot_imp(model):
    feature_imp = pd.Series(
        model.feature_importances_, index=training_cols
    ).sort_values(ascending=False)

    #%matplotlib inline
    # Creating a bar plot
    sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels to your graph
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()


def check_accuracy(model, X_test, y_test):
    # plot_imp(model)

    y_pred = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    plt.figure(figsize=(15, 8))
    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    sns.heatmap(conf_mat, annot=True, fmt="g")
    plt.title("Confusion Matrix ", fontsize=14)
    plt.ylabel("Real Class", fontsize=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.show()


def generate_submission(model, training_cols):
    test = pd.read_csv("data/test.csv")

    test["change_perc"] = 0
    test["change_logdiff"] = 0

    X_test, tp = prep_data(test, training_cols)

    y_pred = model.predict(X_test)

    final = pd.read_csv("data/sample_submission.csv")

    final["num_sold"] = y_pred

    final.to_csv("Result.csv", index=False)


def tune_hyperparameters(t_data, t_pred):
    rfc = RandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": [5, 20, 50, 100, 200, 500],
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": [int(x) for x in np.linspace(10, 120, num=12)],
        "criterion": ["gini", "entropy"],
    }

    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    CV_rfc.fit(t_data, t_pred)

    CV_rfc.best_params_


#%%
if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")

    group_cols = ("country", "store", "product")

    df["change_perc"] = (
        (df["num_sold"] / df.groupby(list(group_cols))["num_sold"].shift() - 1)
        .replace([np.inf, -np.inf, np.nan], 0.0)
        .astype(np.float32)
    )

    df["change_logdiff"] = (
        np.log(df["num_sold"] / df.groupby(list(group_cols))["num_sold"].shift())
        .replace([np.inf, -np.inf, np.nan], 0.0)
        .astype(np.float32)
    )

    training_cols = [
        "change_perc",
        "change_logdiff",
        "country_Finland",
        "country_Norway",
        "country_Sweden",
        "store_KaggleMart",
        "store_KaggleRama",
        "product_Kaggle Hat",
        "product_Kaggle Mug",
        "product_Kaggle Sticker",
        "weekday",
        "week_no",
        "month",
        "Weekend",
        "year",
    ]

    t_data, t_pred = prep_data(df, training_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        t_data, t_pred, test_size=0.25, random_state=42
    )

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_test, y_test)

    params = {
        "nthread": 10,
        "max_depth": 14,
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "regression_l1",
        "metric": "mape",
        "num_leaves": 128,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 3.097758978478437,
        "lambda_l2": 2.9482537987198496,
        "verbose": 1,
    }

    model = lgb.train(
        params,
        lgb_train,
        3000,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=50,
        verbose_eval=50,
    )

    # check_accuracy(model, X_test, y_test)

    generate_submission(model, training_cols)

#%%
