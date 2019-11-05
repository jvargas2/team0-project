from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
import torch

class Seqvec(object):
    """Class for using ELMO embeddings to generate features from protein sequences
        https://github.com/mheinzinger/SeqVec

        Example usage below:

        from seqvec import Seqvec
        sv = Seqvec()
        sv.generate_sequence_embedding('PROTEIN') # Tensor with shape [3, L,1024]
        sv.generate_residue_embedding('PROTEIN') # Tensor with shape [L,1024]
        sv.generate_protein_embedding('PROTEIN') # Tensor with shape [1024]
    """

    def __init__(self):
        """Constructor: Initialize ElmoEmbedder
            Weights file is too big for GitHub and needs to be downloaded from https://rostlab.org/~deepppi/seqvec.zip
        """
        model_dir = Path('data/uniref50_v2')
        weights = model_dir / 'weights.hdf5'
        options = model_dir / 'options.json'
        device = 1 if torch.cuda.is_available() else -1
        self.seqvec  = ElmoEmbedder(options,weights,cuda_device=device) 

    def generate_sequence_embedding(self, seq):
        """Generates a sequence embedding with shape [3,L,1024], where L is the length of the input sequence
            If this is too slow on the GPU, we can also batch encode
        """
        embedding = self.seqvec.embed_sentence(list(seq)) 
        return torch.tensor(embedding)

    def generate_residue_embedding(self, seq):
        """Generates a protein embedding as a single vector with shape [L, 1024], where L is the length of the input sequence
            If this is too slow on the GPU, we can also batch encode
        """
        embedding = self.generate_sequence_embedding(seq)
        return torch.tensor(embedding).sum(dim=0) # Tensor with shape [L,1024]

    def generate_protein_embedding(self, seq):
        """Generates a protein embedding as a single vector with shape [1024]
            If this is too slow on the GPU, we can also batch encode
        """
        embedding = self.generate_sequence_embedding(seq)
        return torch.tensor(embedding).sum(dim=0).mean(dim=0)