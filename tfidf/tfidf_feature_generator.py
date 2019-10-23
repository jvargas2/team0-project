import pandas as pd 
import math
import csv

seqs = pd.read_csv("../data/training.csv", header=0) 
N = len(seqs)

# Distinct amino acids in corpus (Combine all sequences to single string)
corpus = ''.join(seqs.protein.tolist())
vocabulary = set(corpus)

def tf_idf(vocabulary, seq, df_counts, N):
    term_counts = {}

    for t in vocabulary:
        term_counts[t] = seq.count(t)
        
    max_f = max(term_counts.values())
    
    for t in vocabulary:
        tf = term_counts[t] / max_f
        df = df_counts[t]
        w = tf * math.log((N/df),2)
        
    return term_counts

#Document frequency counts
df_counts = {}
for t in vocabulary:
    df_counts[t] = 0
    
    # Count document frequency for each term in vocabulary
    for seq in seqs.protein:
        c = seq.count(t)
        if (c > 0):
            df_counts[t] += 1

writer = None
with open('tfidf_weights.txt', 'w', newline='') as csvfile:
    for ix, seq in seqs.iterrows():   
        features = tf_idf(vocabulary, seq[0], df_counts, N)
        features['ID'] = ix
        features['Label'] = seq[1]
        features['IsDNA'] = 'IsDNA' if seq[1] in ('DNA', 'DRNA') else 'NotDNA'
        features['IsRNA'] = 'IsRNA' if seq[1] in ('RNA', 'DRNA') else 'NotRNA'
        
        # Write headers on first iteration
        if writer is None:
            writer = csv.DictWriter(csvfile, fieldnames=features.keys())
            writer.writeheader()
            
        writer.writerow(features)