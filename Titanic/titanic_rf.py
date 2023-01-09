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
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

# %%
def prep_data(df, training_cols):
    train = df.copy()

    # Extracting Title From Passenger Names
    train["Status"] = train["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

    """df['Status'] = 'Something'
    for i in range(len(df)):
        for x in df.loc[i, 'Name'].split(' '):
            if '.' in x:
                df.loc[i, 'Status'] =  x[:-1]"""

    stats = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4}
    train["Status"] = train.Status.map(stats).fillna(5).astype(int)

    # Gender Preprocess
    gender = {"male": 1, "female": 0}
    train["Sex"] = train["Sex"].replace(gender)

    train_male = train[train["Sex"] == 1]
    train_female = train[train["Sex"] == 0]

    # Fixing Missing Values on Age based on Gender and create bins
    train["Age"].fillna(train.groupby("Sex")["Age"].transform("median"), inplace=True)

    train.loc[train["Age"] <= 12, "Age"] = 0
    train.loc[(train["Age"] > 12) & (train["Age"] <= 18), "Age"] = 1
    train.loc[(train["Age"] > 18) & (train["Age"] <= 28), "Age"] = 2
    train.loc[(train["Age"] > 28) & (train["Age"] <= 38), "Age"] = 3
    train.loc[(train["Age"] > 38) & (train["Age"] <= 48), "Age"] = 4
    train.loc[(train["Age"] > 48) & (train["Age"] <= 58), "Age"] = 5
    train.loc[train["Age"] > 58, "Age"] = 6

    # Embarked PreProcess
    train["Embarked"].fillna("S", inplace=True)
    emb = {"S": 1, "C": 2, "Q": 3}
    train["Embarked"] = train["Embarked"].replace(emb)

    # Fare Price Binning
    train.loc[train["Fare"] == 0, "Fare"] = np.nan

    train["Fare"].fillna(
        train.groupby("Pclass")["Fare"].transform("median"), inplace=True
    )

    train.loc[train["Fare"] <= 8, "Fare"] = 0
    train.loc[(train["Fare"] > 8) & (train["Fare"] <= 18), "Fare"] = 1
    train.loc[(train["Fare"] > 18) & (train["Fare"] <= 50), "Fare"] = 2
    train.loc[(train["Fare"] > 50) & (train["Fare"] <= 100), "Fare"] = 3
    train.loc[train["Fare"] > 100, "Fare"] = 4

    # Binning Ticket Types
    train["Ticket_type"] = train["Ticket"].apply(lambda x: x[0:3])
    train["Ticket_type"] = train["Ticket_type"].astype("category")
    train["Ticket_type"] = train["Ticket_type"].cat.codes

    # Family Size
    train["Family_size"] = train["SibSp"] + train["Parch"] + 1

    # Pclass One hot encoding
    train = pd.get_dummies(train, columns=["Pclass"])

    # train.dropna(inplace=True)
    try:
        t_pred = train["Survived"]
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
    plot_imp(model)

    y_pred = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    plt.figure(figsize=(15, 8))
    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    sns.heatmap(conf_mat, annot=True, fmt="g")
    plt.title("Confusion Matrix of the Random Forest Classifier", fontsize=14)
    plt.ylabel("Real Class", fontsize=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.show()


def generate_submission(model, training_cols):
    test = pd.read_csv("test.csv")
    X_test, tp = prep_data(test, training_cols)

    y_pred = model.predict(X_test)

    final = pd.read_csv("gender_submission.csv")

    final["Survived"] = y_pred

    final.to_csv("RandomForest.csv", index=False)


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
    df = pd.read_csv("train.csv")

    training_cols = [
        "Pclass_1",
        "Pclass_2",
        "Pclass_3",
        "Sex",
        "Age",
        "Family_size",
        "Fare",
        "Status",
    ]

    t_data, t_pred = prep_data(df, training_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        t_data, t_pred, test_size=0.33, random_state=42
    )

    """model = CatBoostClassifier(iterations=15,
                            depth=10,
                            learning_rate=0.2,
                            loss_function='Logloss',
                            verbose=True)"""

    model = RandomForestClassifier(
        max_depth=10, max_features="log2", n_estimators=200, criterion="gini"
    )

    model.fit(X_train, y_train)

    check_accuracy(model, X_test, y_test)

    generate_submission(model, training_cols)
