from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
import torch
import preprocessing
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

def write_to_files(df, seqvec, output_file):
    """Write SeqVec features to files. Creates two files:
    seqvec_1024.csv - contains 1024 feature values for each unique sequence
    seqvec_L_1024.pt - stacked tensors for all sequences
    """
    # Sort proteins by length
    idxs = df.protein.str.len().sort_values().index

    # Create embeddings
    seqs = [list(s.protein) for i, s in df.iloc[idxs].iterrows()]
    embeddings = list(seqvec.embed_sentences(seqs))

    # Revert to original order
    sorted_df = df.iloc[idxs].copy()
    sorted_df.insert(2, 'idx', idxs)
    sorted_df.insert(3, 'embedding', embeddings)
    sorted_df.sort_values('idx', inplace=True)
    df = sorted_df.copy()

    # Write seqvec_1024.csv feature vector
    with open(output_file, 'w', newline='') as csvfile:
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

    # # Write [max(L), N, 1024] tensor
    # tensor_list = [torch.tensor(record.embedding).sum(dim=0) for ix, record in df.iterrows()]
    # stacked_tensor = pad_sequence(tensor_list)
    # torch.save(stacked_tensor, 'data/features/seqvec_L_1024.pt')

def main():
    """
        Uses ELMO embeddings to generate features from protein sequences
        MUST PAD SEQUENCES TO IDENTICAL LENGTH BEFORE USING
        https://github.com/mheinzinger/SeqVec

        Weights file is too big for GitHub and needs to be downloaded from https://rostlab.org/~deepppi/seqvec.zip
        
        Write SeqVec features to files. Creates two files:
        1) seqvec_1024.csv - contains 1024 feature values for each unique sequence
        2) seqvec_N_MAXL_1024.pt - stacked tensor for all sequences with shape [max(L), N, 1024] where N=# of sequences and L=length of each sequence
    """
    # Initialize ElmoEmbedder
    print("Initialize ELMO")
    model_dir = Path('data/uniref50_v2')
    weights = model_dir / 'weights.hdf5'
    options = model_dir / 'options.json'
    device = 1 if torch.cuda.is_available() else -1
    seqvec  = ElmoEmbedder(options,weights,cuda_device=device) 
    
    # Write SeqVec features to files
    print("Load dataframe")
    dataset = "sequences_test"
    input_file = "data/%s.csv" % dataset
    output_file = "data/features/seqvec_%s.csv" % dataset
    print("Running seqvec.py over %s" % dataset)
    df = pd.read_csv(input_file)

    print("Write features to disk")
    write_to_files(df, seqvec, output_file)

    # # Reload
    # loaded = torch.load('data/features/seqvec_L_1024.pt')
    # print('SHAPE', loaded.shape)

if __name__ == "__main__":
    main()