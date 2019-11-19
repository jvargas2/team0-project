import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

def initialize(alphabet):
    """Initialize the amino acid feature dictionary"""
    features = {}
    header = []

    for aa in alphabet:
        features[aa] = []

    i = 2
    aaindex = open("data/aaindex1", "r")
    for line in aaindex:            
        if line[0] == 'H':
            accession_num = line.split()[1]
            header.append(accession_num)
        elif line[0] == 'I':
            i = 0
            continue
        
        if i == 0:
            values = line.split()
            features['A'].append(float(values[0]))
            features['R'].append(float(values[1]))
            features['N'].append(float(values[2]))
            features['D'].append(float(values[3]))
            features['C'].append(float(values[4]))
            features['Q'].append(float(values[5]))
            features['E'].append(float(values[6]))
            features['G'].append(float(values[7]))
            features['H'].append(float(values[8]))
            features['I'].append(float(values[9]))
            i = 1
        elif i == 1:
            values = line.split()
            features['L'].append(float(values[0]))
            features['K'].append(float(values[1]))
            features['M'].append(float(values[2]))
            features['F'].append(float(values[3]))
            features['P'].append(float(values[4]))
            features['S'].append(float(values[5]))
            features['T'].append(float(values[6]))
            features['W'].append(float(values[7]))
            features['Y'].append(float(values[8]))
            features['V'].append(float(values[9]))
            i = 2
    return features, header

def write_to_csv(df, alphabet, header, features, output_file):
    """Write AAIndex features to CSV"""
    feature_cnt = len(header)

    with open(output_file, 'w', newline='') as csvfile:
        # Write header row
        fields = '\t'.join(map(str,header))
        csvfile.write(fields)
        
        # Write protein features
        for ix, record in df.iterrows():
            csvfile.write("\n")
            x = [torch.tensor(features[aa]) if aa in alphabet else torch.zeros(feature_cnt) for aa in record.protein]
            row = torch.stack(x, dim=0).mean(dim=0).numpy()
            row = '\t'.join(map(str, row))
            csvfile.write(row)   

def write_to_file(df, alphabet, features):
    """Write AAIndex features to a file. 
    All sequences are zero padded at the end to make them identical lengths.

    aaindex_MAXL_553.pt - contains a 2D tensor for all sequences of shape [N, max(L), 553]
    where L is the length of each protein, and 553 is the # of AAIndex features
    """   
    # Create tensors
    tensor_list = []
    feature_cnt = len(features['A'])
    for seq in df.protein:
        x = [torch.tensor(features[aa]) if aa in alphabet else torch.zeros(feature_cnt) for aa in seq]
        protein_tensor = torch.stack(x, dim=0)
        tensor_list.append(protein_tensor)

    # Save a tensor of shape [N, max(L), 553].
    # Where N is the # of proteins, L is the length of each protein, and 553 is the # of AAIndex features
    stacked_tensor = pad_sequence(tensor_list)
    torch.save(stacked_tensor, 'data/features/aaindex_MAXL_N_553.pt')
    print('SHAPE', stacked_tensor.shape)

def main():
    """Class for encoding biophysical features for amino acids from the AAIndex
        Writes two files to disk:
        1) aaindex.csv - CSV file with 553 feature columns
        2) aaindex_MAXL_N_553.pt - contains a 2D tensor for all sequences of shape [max(L), N, 553]
            where L is the length of each protein, and 553 is the # of AAIndex features
    """
    dataset = "sequences_test"
    input_file = "data/%s.csv" % dataset
    output_file = "data/features/aaindex_%s.csv" % dataset
    print("Running aaindex.py over %s" % dataset)
    alphabet = ['A', 'L', 'R', 'K', 'N', 'M', 'D', 'F', 'C', 'P', 'Q', 'S', 'E', 'T', 'G', 'W', 'H', 'Y', 'I', 'V']
    features, header = initialize(alphabet)
    df = pd.read_csv(input_file)

    print("Write features to CSV")
    write_to_csv(df, alphabet, header, features, output_file)

    # print("Write 2D tensor to file")
    # write_to_file(df, alphabet, features)

if __name__ == "__main__":
    main()