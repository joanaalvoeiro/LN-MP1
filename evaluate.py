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

    matches = 0
    n_lines = len(lines1)
    for i in range(n_lines):
        if(lines1[i] == lines2[i]):
            matches += 1
    
    print("Precision = {}%".format(matches/n_lines * 100))

if __name__ == "__main__":
    main()