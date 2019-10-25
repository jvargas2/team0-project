import math

class Tfidf(object):
    """Class for encoding one-dimensional tf-idf weight vectors for amino acid sequences
        Must be initialized by passing in the list of all sequences
        Example usage below:

        from tfidf import Tfidf
        import preprocessing
        df = preprocessing.load_df()
        tfidf_feature_generator = Tfidf(list(df.protein))
        tfidf_feature_generator.encode_sequence('AFD')
    """

    def __init__(self, sequences):
        """Constructor: Initialize objects to be used with each encoding"""
        # Number of sequences in the corpus
        self.N = len(sequences)

        # Unique characters (amino acids) in the corpus
        self.vocabulary = list(set(''.join(sequences)))
        self.vocabulary.sort()

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
