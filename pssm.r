# Install R, R Studio, and NCBI Blast++ (https://www.blaststation.com/intl/members/en/howtoblastwin.html)
# Package Documentation (https://cran.r-project.org/web/packages/protr/vignettes/protr.html)
# GitHub (https://github.com/nanxstats/protr/blob/master/R/desc-15-PSSM.R)
# Install packages
install.packages("protr")
install.packages("BiocManager")
BiocManager::install("Biostrings")


library("protr")

# Set the working directory path if necessary
#setwd("<PATH/TO/DATA.fasta")

# Load protein sequences from fasta file
# Example FASTA format: (first line just an > and some dummy identifier, second line amino acid sequence)
# > seq1
# ADFEWEWRGF
seqs <- readFASTA("dummy.fasta")
length(seqs)

# Remove sequences with Amino Acids other than 20 most common
seqs  <- seqs[(sapply(seqs, protcheck))]
length(seqs)

# This results in a list of PSSM matrices corresponsing to the sequences in the fasta file of shape [20, L] where L is the length of the sequence
pssm <- t(sapply(seqs, extractPSSM, 
    makeblastdb.path="C:/Program Files/NCBI/blast-2.7.1+/bin/makeblastdb", 
    psiblast.path="C:/Program Files/NCBI/blast-2.7.1+/bin/psiblast",
    database.path="dummy.fasta"))