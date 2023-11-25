import csv
import sys

def reduce_f(lines_file, users_file, out, delimiter="\t"):
    users = set()
    with open(users_file) as rec_file:
        tsv_file = csv.reader(rec_file, delimiter="\t")
        for line in tsv_file:
            users.add(line[0])
    with open(lines_file) as rec_file:
        tsv_file = csv.reader(rec_file, delimiter=delimiter)
        writer = csv.writer(open(out,"w"), delimiter=delimiter)
        for line in tsv_file:
            if line[0] in users:
                writer.writerow(line)

if __name__ == '__main__':
    delimiter = "\t"
    if len(sys.argv) == 5 and sys.argv[4] == "S":
        delimiter = " "

    reduce_f(sys.argv[1], sys.argv[2], sys.argv[3], delimiter)
