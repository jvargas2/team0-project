import os
import pandas as pd

def load_df(name='training.csv'):
    cwd = os.path.dirname(os.path.abspath(__file__))
    dataset_name = cwd + '/data/%s' % name
    df = pd.read_csv(dataset_name)
    return df

def get_counts(df):
    count_df = df.groupby('class').count()
    count_df = count_df.append([{'protein': df.count()[0]}])
    print(count_df)

def example_model(df):
    df['prediction'] = 'nonDRNA'
    return df

def evaluate(df):
    if 'prediction' not in df.columns:
        raise ValueError('Needs a prediction column to evaluate')

if __name__ == '__main__':
    df = load_df()
    get_counts(df)
    df = example_model(df)
    evaluate(df)
