import os
from math import sqrt
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

    labels = ['DNA', 'RNA', 'DRNA', 'nonDRNA']
    label_results = {}

    for label in labels:
        tp = df.get(df['class'] == label).get(df['prediction'] == label).count()[0].item()
        tn = df.get(df['class'] != label).get(df['prediction'] != label).count()[0].item()
        fp = df.get(df['class'] != label).get(df['prediction'] == label).count()[0].item()
        fn = df.get(df['class'] == label).get(df['prediction'] != label).count()[0].item()

        total = tp + tn + fp + fn
        accuracy = 100 * (tp + tn) / total
        sensitivity = 100 * tp / (tp + fn)
        specificity = 100 * tn / (tn + fp)

        denominator = sqrt((tp + fn) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator == 0:
            denominator = 1
        mcc = (tp * fn - fp * fn) / denominator
        
        label_results[label] = {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'mcc': mcc
        }

    for label, results in label_results.items():
        print('----%s----' % label)
        for key, value in results.items():
            print('  %s: %f' % (key, value))

def evaluate_file(filename):
    df = pd.load_csv(filename)
    evaluate(df)

if __name__ == '__main__':
    df = load_df()
    get_counts(df)
    df = example_model(df)
    evaluate(df)
