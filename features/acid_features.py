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
        drna_polarities = []
        drna_h = []
        non_drna_polarities = []
        non_drna_h = []
        for line in csv_reader:
            polarity = get_overall_polarity(line[0])
            hydrophobicity = get_overall_hydrophobicity(line[0])
            if line[1] == "1":
                drna_polarities.append(polarity)
                drna_h.append(hydrophobicity)
            else:
                non_drna_polarities.append(polarity)
                non_drna_h.append(hydrophobicity)
        with open(sys.argv[2], "w") as output_csv:
            csv_writer = csv.writer(output_csv)
            csv_writer.writerow(["Hydrophobicity", "Polarity", "Class"])
            for i in range(len(drna_polarities)):
                csv_writer.writerow([drna_h[i], drna_polarities[i], 1])
            for i in range(len(non_drna_polarities)):
                csv_writer.writerow([non_drna_h[i], non_drna_polarities[i], 0])


if __name__ == "__main__":
    main()
