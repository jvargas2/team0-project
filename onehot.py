class Onehot(object):
    """Class for encoding two dimensional onehot feature vectors for amino acid sequences
        Must be initialized by passing in the 'alphabet' (list of unique characters) to be used for the encoding.
        Example usage below:

        from onehot import Onehot
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        onehot_feature_generator = Onehot(alphabet)
        onehot_feature_generator.encode_sequence('CADZSDFDFGSDFGADFGSADFASDGF')
    """

    def __init__(self, alphabet):
        """Constructor: Initialize objects to be used with each encoding"""
        self.alphabet = alphabet
        self.char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        self.zero_vector = range(len(self.alphabet))

    # Code appropriated from https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    def encode_sequence(self, seq):
        """Method to encode a onehot vector from a given sequence
            TODO: This is currently NOT zero-padding vectors. Handle this here or by caller?
        """
        # integer encode input data
        integer_encoded = [self.char_to_int[char] for char in seq]

        # one hot encode
        onehot_2d_vector = list()
        for value in integer_encoded:
            letter = [0 for _ in self.zero_vector]
            letter[value] = 1
            onehot_2d_vector.append(letter)

        return onehot_2d_vector