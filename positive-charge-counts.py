import csv 

with open('./data/training_dna.csv', 'r') as dna_file:
    csv_reader = csv.reader(dna_file)
    line_num = 0
    dna = []
    non = []
    for line in csv_reader:
        if line_num > 0:
            sequence = line[0]
            pos_count = 0.0
            total_aa = 0.0
            for letter in sequence:
                if letter in ['H', 'K', 'R']:
                    pos_count += 1
                total_aa += 1
            if int(line[1]) == 1:
                dna.append(pos_count / total_aa)
            else:
                non.append(pos_count / total_aa)
        line_num += 1
    print(sum(dna) / len(dna))
    print(sum(non) / len(non))