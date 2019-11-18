import argparse
from math import sqrt

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import multilabel_confusion_matrix

from character_bilstm import CharacterBiLSTM
from bilstm import BiLSTM
from feature_linear import FeatureLinear
from proteins_dataset import ProteinsDataset
from quad_linear import QuadLinear

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', help='Run with smaller dataset for debugging')
    parser.add_argument('-g', '--gpu', default=None, type=int, help='GPU device to use')
    parser.add_argument('-f', '--features', default='acid', help='Which features to use')

    args = parser.parse_args()

    torch.manual_seed(0)
    dna_dataset = ProteinsDataset(debug=args.debug, features=args.features, labels={'DNA': 1, 'RNA': 0, 'DRNA': 1, 'nonDRNA': 0})
    rna_dataset = ProteinsDataset(debug=args.debug, features=args.features, labels={'DNA': 0, 'RNA': 1, 'DRNA': 1, 'nonDRNA': 0})
    n_splits = 2 if args.debug else 5
    max_epochs = 1 if args.debug else 1000
    gpus = None if args.gpu is None else [args.gpu]
    num_features = None
    batch_size = 1
    model_class = CharacterBiLSTM

    if args.features == 'acid':
        num_features = 2
        model_class = FeatureLinear
    elif args.features == 'character':
        model_class = CharacterBiLSTM
    elif args.features == 'protvec':
        num_features = 100
        model_class = FeatureLinear
    elif args.features == 'count':
        num_features = 23
        model_class = FeatureLinear
    elif args.features == 'onehot':
        num_features = 23
        batch_size = 30 if self.on_gpu else 5
        model_class = BiLSTM
    elif args.features == 'aaindex':
        num_features = 553
        model_class = FeatureLinear
    elif args.features == 'aaindex2d':
        num_features = 553
        batch_size = 1
        model_class = BiLSTM
    elif args.features == 'aaindex-seqvec':
        num_features = 1577
        model_class = FeatureLinear
    else:
        raise ValueError('Invalid features')

    skf = StratifiedKFold(n_splits=n_splits)
    y_true = []
    y_pred = []

    for train_indices, test_indices in skf.split(dna_dataset.x, dna_dataset.y):
        dna_model = model_class(dna_dataset, train_indices, test_indices, num_features, batch_size)
        rna_model = model_class(rna_dataset, train_indices, test_indices, num_features, batch_size)
        dna_trainer = Trainer(max_nb_epochs=max_epochs, gpus=gpus)
        rna_trainer = Trainer(max_nb_epochs=max_epochs, gpus=gpus)
        dna_trainer.fit(dna_model)
        rna_trainer.fit(rna_model)

        for i in test_indices:
            protein, label = dataset[i]
            protein = protein.unsqueeze(0)
            if args.gpu is not None:
                protein = protein.cuda(device='cuda:%d' % args.gpu)
            dna_output = dna_model(protein)
            rna_output = rna_model(protein)
            dna_prediction = torch.max(dna_output, 1)[1].item()
            rna_prediction = torch.max(rna_output, 1)[1].item()

            prediction = None

            if dna_prediction == 1 and rna_prediction == 1:
                prediction = 2
            elif dna_prediction == 1:
                prediction = 0
            elif rna_prediction == 1:
                prediction = 1
            elif dna_prediction == 0 and rna_prediction == 0:
                prediction = 3
            else:
                raise ValueError('Something went wrong.')

            y_true.append(label)
            y_pred.append(prediction)

    labels = ['DNA', 'RNA', 'DRNA', 'nonDRNA']
    label_results = {}

    confusion_matrix = multilabel_confusion_matrix(y_true, y_pred, labels=[0,1,2,3])

    for index, label in enumerate(labels):
        cm = confusion_matrix[index]
        tp = cm[1][1]
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]

        total = tp + tn + fp + fn
        accuracy = 100 * (tp + tn) / total
        sensitivity = 100 * tp / (tp + fn)
        specificity = 100 * tn / (tn + fp)

        denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator == 0:
            denominator = 1
        mcc = (tp * tn - fp * fn) / denominator
        
        label_results[label] = {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'mcc': mcc
        }

    average_mcc = sum([label_results[label]['mcc'] for label in labels]) / 4
    average_accuracy = 100 * sum([label_results[label]['tp'] for label in labels]) / len(y_true)

    for label, results in label_results.items():
        print('----%s----' % label)
        for key, value in results.items():
            if key == 'mcc':
                print('  %s: %.3f' % (key, value))
            else:
                print('  %s: %.1f' % (key, value))

    print('----Average----')
    print('  mcc: %.3f' % average_mcc)
    print('  accuracy: %.1f' % average_accuracy)

if __name__ == '__main__':
    main()
