import torch

class AaIndex(object):
    """Class for encoding biophysical features for amino acids from the AAIndex

        Example usage below:

        import preprocessing
        from aaindex import AaIndex

        df = preprocessing.load_df()
        aa_index = AaIndex()

        # Write features to CSV
        aa_index.write_to_csv(df)

        # Write tensor for Bi-LSTM to file
        aa_index.write_to_file(df)
    """

    def __init__(self):
        """Constructor"""
        self.alphabet = ['A', 'L', 'R', 'K', 'N', 'M', 'D', 'F', 'C', 'P', 'Q', 'S', 'E', 'T', 'G', 'W', 'H', 'Y', 'I', 'V']
        self.features, self.header = self.initialize()

    def initialize(self):
        """Initialize the amino acid feature dictionary"""
        features = {}
        header = []

        for aa in self.alphabet:
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

    def write_to_csv(self, df):
        """Write AAIndex features to CSV"""
        with open('data/features/aaindex.csv', 'w', newline='') as csvfile:
            # Write header row
            fields = '\t'.join(map(str,self.header))
            csvfile.write(fields)
            
            # Write protein features
            for ix, record in df.iterrows():
                csvfile.write("\n")
                x = [torch.tensor(aa_index.features[aa]) if aa in aa_index.alphabet else torch.zeros(feature_cnt) for aa in record.protein]
                row = torch.stack(x, dim=0).mean(dim=0).numpy()
                row = '\t'.join(map(str, row))
                csvfile.write(row)   

    def write_to_file(self, df):
        """Write AAIndex features to a file. 
        All sequences are zerro padded at the end to make them identical lengths.

        aaindex_MAXL_553.pt - contains a stacked tensor for all sequences of shape [N, max(L), 553]
        where L is the length of each protein, and 553 is the # of AAIndex features
        """
        # Pad sequences
        feature_cnt = len(self.header)
        max_seq_len = len(max(df.protein, key=len))
        seqs = [protein + '0'*(max_seq_len-len(protein)) for protein in df.protein]

        # Create tensors
        tensor_list = []
        for seq in seqs:
            x = [torch.tensor(self.features[aa]) if aa in self.alphabet else torch.zeros(feature_cnt) for aa in seq]
            protein_tensor = torch.stack(x, dim=0)
            tensor_list.append(protein_tensor)

        # Save a tensor of shape [N, max(L), 553].
        # Where N is the # of proteins, L is the length of each protein, and 553 is the # of AAIndex features
        stacked_tensor = torch.stack(tensor_list)
        torch.save(stacked_tensor, 'data/features/aaindex_MAXL_553.pt')
