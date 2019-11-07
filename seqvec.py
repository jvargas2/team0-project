from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
import torch

class Seqvec(object):
    """Class for using ELMO embeddings to generate features from protein sequences
        MUST PAD SEQUENCES TO IDENTICAL LENGTH BEFORE USING
        https://github.com/mheinzinger/SeqVec

        Example usage below:

        import preprocessing
        from seqvec import Seqvec
        import torch

        # Initialize
        df = preprocessing.load_df()
        sv = Seqvec()

        # NOTE: Sequences should all be zero padded to the same length before running this

        # Write SeqVec features to files. Creates two files:
        #  seqvec_1024.csv - contains 1024 feature values for each unique sequence
        #  seqvec_L_1024.pt - stacked tensor for all sequences with shape [N, L, 1024]
        #   where N=# of sequences and L=length of each sequence

        sv.write_to_files(df)

        # Reload the stacked tensor from file
        loaded = torch.load('data/features/seqvec_L_1024.pt')
        loaded.shape
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

    def write_to_files(self, df):
        """Write SeqVec features to files. Creates two files:
        seqvec_1024.csv - contains 1024 feature values for each unique sequence
        seqvec_L_1024.pt - stacked tensors for all sequences
        """

        # Sort proteins by length
        idxs = df.protein.str.len().sort_values().index

        # Create embeddings
        seqs = [list(s.protein) for i, s in df.iloc[idxs].iterrows()]
        embeddings = list(self.seqvec.embed_sentences(seqs))

        # Revert to original order
        sorted_df = df.iloc[idxs].copy()
        sorted_df.insert(2, 'idx', idxs)
        sorted_df.insert(3, 'embedding', embeddings)
        sorted_df.sort_values('idx', inplace=True)
        df = sorted_df.copy()

        # Write seqvec_1024.csv feature vector
        with open('data/features/seqvec_1024.csv', 'w', newline='') as csvfile:
            # Write header row
            fields = [str(s) + 'd_seqvec' for s in range(0, 1024)]
            fields = ['protein'] + fields
            header = '\t'.join(map(str, fields))
            csvfile.write(header)

            # Write protein features
            for ix, record in df.iterrows():
                csvfile.write("\n")
                features = torch.tensor(record.embedding).sum(dim=0).mean(dim=0).numpy()  # Vector with shape [1024]
                row = '\t'.join(map(str, features))
                csvfile.write(record.protein + '\t')
                csvfile.write(row)

        # Write [L,1024] tensor
        tensor_list = [torch.tensor(record.embedding).sum(dim=0) for ix, record in df.iterrows()]
        stacked_tensor = torch.stack(tensor_list)
        torch.save(stacked_tensor, 'data/features/seqvec_L_1024.pt')