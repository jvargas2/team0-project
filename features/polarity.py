import csv
import sys

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


def main():
    input_file = sys.argv[1]
    with open(input_file, "r") as input_csv:
        csv_reader = csv.reader(input_csv)
        dna_polarities = []
        non_dna_polarities = []
        for line in csv_reader:
            polarity = get_overall_polarity(line[0])
            if line[1] == "1":
                dna_polarities.append(polarity)
            else:
                non_dna_polarities.append(polarity)
        with open(sys.argv[2], "w") as output_csv:
            csv_writer = csv.writer(output_csv)
            csv_writer.writerow(["Polarity", "Class"])
            for pol in dna_polarities:
                csv_writer.writerow([pol, 1])
            for pol in non_dna_polarities:
                csv_writer.writerow([pol, 0])


if __name__ == "__main__":
    main()
