import argparse
from math import sqrt

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import multilabel_confusion_matrix

from character_bilstm import CharacterBiLSTM
from proteins_dataset import ProteinsDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', help='Run with smaller dataset for debugging')
    parser.add_argument('-g', '--gpu', default=None, type=int, help='GPU device to use')

    args = parser.parse_args()

    dataset = ProteinsDataset(debug=args.debug)
    n_splits = 2 if args.debug else 5
    max_epochs = 1 if args.debug else 30
    gpus = None if args.gpu is None else [args.gpu]

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=3,
        verbose=True,
        mode='min'
    )

    skf = StratifiedKFold(n_splits=n_splits)
    y_true = []
    y_pred = []

    for train_indices, test_indices in skf.split(dataset.x, dataset.y):
        model = CharacterBiLSTM(dataset, train_indices, test_indices)
        trainer = Trainer(max_nb_epochs=max_epochs, gpus=gpus, early_stop_callback=early_stop_callback)
        trainer.fit(model)

        for i in test_indices:
            protein, label = dataset[i]
            protein = protein.unsqueeze(0)
            if args.gpu is not None:
                protein = protein.cuda(device='cuda:%d' % args.gpu)
            output = model(protein)
            prediction = torch.max(output, 1)[1].item()
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
