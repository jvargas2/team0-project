import math

class Tfidf(object):
    """Class for encoding one-dimensional tf-idf weight vectors for amino acid sequences
        Must be initialized by passing in the list of all sequences
        Example usage below:

        from tfidf import Tfidf
        import preprocessing
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        df = preprocessing.load_df()
        tfidf_feature_generator = Tfidf(list(df.protein), alphabet)
        tfidf_feature_generator.encode_sequence('AFD')
        tfidf_feature_generator.write_to_csv(df)
    """

    def __init__(self, sequences, vocabulary):
        """Constructor: Initialize objects to be used with each encoding"""
        # Number of sequences in the corpus
        self.N = len(sequences)

        # Unique characters (amino acids) in the corpus
        self.vocabulary = vocabulary

        # Dictionary of document frequency (DF) counts for each term in the vocabulary
        self.df_counts = self.initialize(sequences)

    def initialize(self, sequences):
        """Calculate document frequency (DF) counts for each term (amino acid) in the vocabulary"""
        df_counts = {}
        for t in self.vocabulary:
            df_counts[t] = 0

            # Count document frequency for each term in vocabulary
            for seq in sequences:
                c = seq.count(t)
                if c > 0:
                    df_counts[t] += 1
        return df_counts

    def encode_sequence(self, seq):
        """Method to encode a one-dimensional feature vector of TF-IDF weights for a given sequence"""
        term_counts = {}
        weights = []

        for t in self.vocabulary:
            term_counts[t] = seq.count(t)

        max_f = max(term_counts.values())

        for t in self.vocabulary:
            tf = term_counts[t] / max_f
            df = self.df_counts[t]
            w = tf * math.log((self.N / df), 2)
            weights.append(w)

        return weights

    def write_to_csv(self, df, features_only = True):
        """Write TF-IDF features to CSV"""
        with open('data/features/tfidf.csv', 'w', newline='') as csvfile:
            # Write header row
            fields = [s + '_tfidf' for s in self.vocabulary]
            if not features_only: 
                fields = ['class', 'protein'] + fields
            header = '\t'.join(map(str,fields))
            csvfile.write(header)
            
            # Write protein features
            for ix, record in df.iterrows():     
                csvfile.write("\n")
                features = self.encode_sequence(record.protein)
                if not features_only: 
                    features = [record['class'], record.protein] + features
                row = '\t'.join(map(str, features))
                csvfile.write(row)

