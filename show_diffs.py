import sys

def main():
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    with open("DEV-questions.txt") as qf:
        questions = qf.readlines()
    qf.close()

    with open("treated_known_questions.txt") as known:
        known_questions = known.readlines()
    known.close()

    with open("treated_test_questions.txt") as test:
        test_questions = test.readlines()
    test.close()

    with open(file1) as f1:
        lines1 = f1.readlines()
    f1.close()

    with open(file2) as f2:
        lines2 = f2.readlines()
    f2.close()

    matches = 0
    n_lines = len(lines1)
    for i in range(n_lines):
        if(lines1[i] != lines2[i]):
            print("-------------------------------------------------------")
            print(questions[i])
            print("Supposed to be: {}".format(lines1[i]))
            print("Was: {}".format(lines2[i]))
            print("Tokenized as: {}".format(test_questions[i]))
            print("Most similar question was: {}".format(known_questions[i]))

if __name__ == "__main__":
    main()