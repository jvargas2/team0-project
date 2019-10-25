import numpy as np
import csv

class Protvec(object):
    """Class for encoding a 100 feature vector for a given sequence using pretrained embeddings.
        The pre-trained embeddings were obtained from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JMFHTN
        The resulting vector is the average of the embedding vectors for all 3-grams in the input sequence

        Example usage below:

        from protvec import Protvec
        pv = Protvec()
        pv.encode_sequence('CADZSDFDFGSDFGADFGSADFASDGF')
    """

    def __init__(self):
        """Constructor: Initialize objects to be used with each encoding"""
        self.embeddings_index = self.initialize_embeddings_index()

    def initialize_embeddings_index(self):
        """"Intialize the dictionary of embeddings from a file of pre-trained embeddings"""
        embeddings_index = {}
        with open('data/protVec_100d_3grams.csv') as file:
            reader = csv.reader(file, delimiter='\t')
            next(reader)  # skip header

            for line in reader:
                word = line[0]
                coefs = np.asarray(line[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index

    def ngrams(self, seq, n):
        """Generates all ngrams of length N in the sequence"""
        output = []
        for i in range(len(seq) - n + 1):
            output.append(seq[i:i + n])
        return output

    def encode_sequence(self, seq):
        """Averages the embedding vector for each 3gram in the given sequence into a single feature vector"""
        # Kmers is a subsequence of length k=3
        kmers = self.ngrams(seq, 3)

        kmer_embeddings = []
        for kmer in kmers:
            if kmer in self.embeddings_index:
                kmer_embeddings.append(self.embeddings_index[kmer])

        return np.mean(np.array(kmer_embeddings), axis=0)