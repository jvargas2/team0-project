import csv

dna_sequences = []
rna_sequences = []
nonDRNA_sequences = []
with open('./data/training.csv', 'r') as training_file:
    csv_reader = csv.reader(training_file)
    line_number = 0
    for line in csv_reader:
        if line_number > 0:
            if line[1] == 'DRNA':
                dna_sequences.append([line[0]])
                rna_sequences.append([line[0]])
            elif line[1] == 'DNA':
                dna_sequences.append([line[0]])
            elif line[1] == 'RNA':
                rna_sequences.append([line[0]])
            else:
                nonDRNA_sequences.append([line[0]])
        line_number += 1
with open('./data/training_dna.csv', 'w') as dna_file:
    csv_writer = csv.writer(dna_file)
    for sequence in dna_sequences:
        csv_writer.writerow(sequence)