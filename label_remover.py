import sys

filename = sys.argv[1]

coarse_labels = []
fine_labels = []
questions = []

with open(filename) as f:
    lines = f.readlines()
f.close()

for line in lines:
    question_index = line.find(" ")
    fine_index = line.find(":")

    coarse_labels.append(line[:fine_index] + "\n")
    fine_labels.append(line[:question_index] + "\n")
    questions.append(line[question_index+1:])

labels_file = filename[:-4] + "-labels.txt"
questions_file = filename[:-4] + "-questions.txt"


with open(labels_file, 'w') as lf:
    with open(questions_file, 'w') as qf:
        for i in range(len(lines)):
            lf.write(fine_labels[i])
            qf.write(questions[i])

qf.close()
lf.close()