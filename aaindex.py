import torch

class AaIndex(object):
    """Class for encoding biophysical features for amino acids from the AAIndex

        Example usage below:

        import preprocessing
        from aaindex import AaIndex

        df = preprocessing.load_df()
        aa_index = AaIndex()
        aa_index.write_to_csv(df)
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
                x = [torch.tensor(self.features[aa]) for aa in record.protein if aa in self.alphabet]
                row = torch.stack(x, dim=0).mean(dim=0).numpy()
                row = '\t'.join(map(str, row))
                csvfile.write(row)   

