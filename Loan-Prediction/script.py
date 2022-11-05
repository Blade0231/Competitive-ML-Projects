import pandas as pd

def preprocess_pipeline(df):
    return df


loan_data = pd.read_csv('data/train.csv')

loan_data = preprocess_pipeline(df=loan_data)
