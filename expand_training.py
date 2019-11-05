import csv

dna_sequences = []
rna_sequences = []
both_sequences = []
with open("./DNA_RNA_BIND.txt", "r") as input_file:
    current_label = ""
    for line in input_file:
        if line[0] == ">":
            current_label = line.split()[1]
        else:
            if current_label == "RNA_bind":
                if line in dna_sequences:
                    both_sequences.append(line)
                rna_sequences.append(line)
            else:
                if line in rna_sequences:
                    both_sequences.append(line)
                dna_sequences.append(line)

old_sequences = []
old_lines = []
old_sequences_all = []
old_lines_all = []

with open("./data/training.csv", "r") as input_file:
    csv_reader = csv.reader(input_file)
    for line in csv_reader:
        old_sequences_all.append(line[0])
        old_lines_all.append(line)

# with open("./data/training_dna.csv", "r") as input_file:
#     csv_reader = csv.reader(input_file)
#     for line in csv_reader:
#         old_sequences.append(line[0])
#         old_lines.append(line)

# with open("./data/training_dna_expanded.csv", "w") as output_file:
#     csv_writer = csv.writer(output_file)
#     for line in old_lines:
#         csv_writer.writerow(line)
#     for sequence in dna_sequences:
#         if not sequence in old_sequences:
#             csv_writer.writerow([sequence, 1])

# with open("./data/training_rna_expanded.csv", "w") as output_file:
#     csv_writer = csv.writer(output_file)
#     for line in old_lines:
#         csv_writer.writerow(line)
#     for sequence in rna_sequences:
#         if not sequence in old_sequences:
#             csv_writer.writerow([sequence, 1])


# with open("./data/training_rna_expanded.csv", "r") as input_file:
#     csv_reader = csv.reader(input_file)
#     dna_count = 0
#     non_count = 0
#     for line in csv_reader:
#         if line[1] == "0":
#             non_count += 1
#         else:
#             dna_count += 1

with open("./data/training_expanded.csv", "w") as output_file:
    csv_writer = csv.writer(output_file)
    for line in old_lines_all:
        csv_writer.writerow(line)
    for sequence in rna_sequences:
        if not sequence in old_sequences_all:
            if sequence in both_sequences:
                csv_writer.writerow([sequence, "DRNA"])
            else:
                csv_writer.writerow([sequence, "RNA"])
    for sequence in dna_sequences:
        if not sequence in old_sequences_all:
            if not sequence in both_sequences:
                csv_writer.writerow([sequence, "DNA"])
