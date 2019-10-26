import pandas
import csv

AMINO_ACIDS = ['G', 'P', 'A', 'V', 'L', 'I', 'M', 'C', 'F','Y', 'W', 'H', 'K', 'R', 'Q', 'N', 'E', 'D', 'S', 'T']

def get_aa_counts(sequence):
    counts = {}
    for letter in sequence:
        if letter in counts.keys():
            counts[letter] += 1
        else:
            counts[letter] = 1
    return counts

def print_count_data(data, filename):
    with open(filename, 'w') as output:
        csv_writer = csv.writer(output)
        csv_writer.writerow(AMINO_ACIDS + ['Class'])
        for index, row in data.iterrows():
            counts = get_aa_counts(row['Sequence'])
            output_array = []
            for letter in AMINO_ACIDS:
                if letter in counts.keys():
                    output_array.append(counts[letter])
                else:
                    output_array.append(0)
            output_array.append(row['Class'])
            csv_writer.writerow(output_array)

def main():
    dna_data = pandas.read_csv("./data/training_dna.csv")
    rna_data = pandas.read_csv("./data/training_rna.csv")
    print_count_data(dna_data, './data/aa_counts_dna.csv')
    print_count_data(rna_data, './data/aa_counts_rna.csv')


if __name__ == "__main__":
    main()