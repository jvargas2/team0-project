import pandas
from sklearn import svm
from sklearn.model_selection import cross_val_score


def create_model(data):
    labels = data['Class']
    del data['Class']
    clf = svm.SVC()
    scores = cross_val_score(clf, data, labels, cv=5)
    print(scores)


def main():
    dna_data = pandas.read_csv("./data/hydrophobicity_dna.csv")
    rna_data = pandas.read_csv("./data/hydrophobicity_rna.csv")
    create_model(dna_data)
    create_model(rna_data)


if __name__ == "__main__":
    main()
