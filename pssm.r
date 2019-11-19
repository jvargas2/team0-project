# Install R, R Studio, and NCBI Blast++ (https://www.blaststation.com/intl/members/en/howtoblastwin.html)
# Package Documentation (https://cran.r-project.org/web/packages/protr/vignettes/protr.html)
# GitHub (https://github.com/nanxstats/protr/blob/master/R/desc-15-PSSM.R)
# Install packages


# # Python code to create FASTA file
# import preprocessing
# alphabet = ['A', 'L', 'R', 'K', 'N', 'M', 'D', 'F', 'C', 'P', 'Q', 'S', 'E', 'T', 'G', 'W', 'H', 'Y', 'I', 'V']
# df = preprocessing.load_df()
# with open('data/training.fasta', 'w', newline='') as csvfile:
#     # Write protein features
#     for ix, record in df.iterrows():
#         # Remove amino acids not in alphabet
#         protein = ''.join([aa for aa in record.protein if aa in alphabet])
        
#         csvfile.write("> protein" + str(ix) + "\n")
#         csvfile.write(protein + "\n")  

install.packages("protr")
install.packages("BiocManager")
BiocManager::install("Biostrings")


library("protr")

# Set the working directory path if necessary
setwd("C:/Users/etfrench/code/team0-project/data")

# Load protein sequences from fasta file
# Example FASTA format: (first line just an > and some dummy identifier, second line amino acid sequence)
# > seq1
# ADFEWEWRGF
seqs <- readFASTA("training.fasta")
length(seqs)

# Remove sequences with Amino Acids other than 20 most common
seqs  <- seqs[(sapply(seqs, protcheck))]
length(seqs)

# This results in a list of PSSM matrices corresponsing to the sequences in the fasta file of shape [20, L] where L is the length of the sequence
pssm <- t(sapply(seqs, extractPSSM, 
    makeblastdb.path="C:/Program Files/NCBI/blast-2.7.1+/bin/makeblastdb", 
    psiblast.path="C:/Program Files/NCBI/blast-2.7.1+/bin/psiblast",
    database.path="training.fasta"))

for (ix in 1:length(pssm)) {
    # R arrays are indexed at one, so ix-1 for naming convention to match python dataframe index
    write.table(p, file=paste("features/pssm/seq", ix-1, ".csv", sep = ""), row.names=FALSE, sep="\t", append = FALSE, quote = FALSE)
}
