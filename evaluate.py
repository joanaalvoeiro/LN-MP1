import sys

def main():
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    with open(file1) as f1:
        lines1 = f1.readlines()
    f1.close()

    with open(file2) as f2:
        lines2 = f2.readlines()
    f2.close()

    if(not ':' in lines2[0]):
        lines3 = lines1
        lines1 = []
        for line in lines3:
            fine_separator = line.find(":")

            coarse_lbl = line[:fine_separator] + "\n"

            lines1.append(coarse_lbl)

    matches = 0
    n_lines = len(lines1)
    for i in range(n_lines):
        if(lines1[i] == lines2[i]):
            matches += 1
    
    print("Precision = {}%".format(matches/n_lines * 100))

if __name__ == "__main__":
    main()