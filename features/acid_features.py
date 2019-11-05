# ARG1: path of input csv
# ARG2: path of output csv

import csv
import sys

PHOBIC_ACIDS = ["G", "A", "V", "L", "I", "P", "F", "M", "W", "C", "Y"]
PHILIC_ACIDS = ["R", "N", "D", "Q", "E", "H", "K", "S", "T"]

POSITIVE_ACIDS = ["R", "K", "H"]
NEGATIVE_ACIDS = ["D", "E"]


def get_overall_polarity(sequence):
    overall_polarity = 0
    for acid in sequence:
        if acid in POSITIVE_ACIDS:
            overall_polarity += 1
        elif acid in NEGATIVE_ACIDS:
            overall_polarity -= 1
    return overall_polarity


def get_overall_hydrophobicity(sequence):
    overall_hydrophobicity = 0
    for acid in sequence:
        if acid in PHILIC_ACIDS:
            overall_hydrophobicity += 1
        elif acid in PHOBIC_ACIDS:
            overall_hydrophobicity -= 1
    return overall_hydrophobicity


def main():
    input_file = sys.argv[1]
    with open(input_file, "r") as input_csv:
        csv_reader = csv.reader(input_csv)

        polarities = {}
        hydros = {}

        for line in csv_reader:
            polarity = get_overall_polarity(line[0])
            hydrophobicity = get_overall_hydrophobicity(line[0])
            protein_class = line[1]
            if protein_class in polarities.keys():
                polarities[protein_class].append(polarity)
                hydros[protein_class].append(hydrophobicity)
            else:
                polarities[protein_class] = [polarity]
                hydros[protein_class] = [hydrophobicity]

        with open(sys.argv[2], "w") as output_csv:
            csv_writer = csv.writer(output_csv)
            csv_writer.writerow(["Hydrophobicity", "Polarity", "Class"])

            for key in polarities.keys():
                polarities_by_class = polarities[key]
                hydros_by_class = hydros[key]
                for i in range(len(polarities[key])):
                    csv_writer.writerow(
                        [hydros_by_class[i], polarities_by_class[i], key])


if __name__ == "__main__":
    main()
