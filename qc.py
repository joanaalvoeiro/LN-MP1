import sys
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.metrics.distance import jaccard_distance

class Question:
    def __init__(self, coarse_lbl, fine_lbl, question):
	    self.coarse_lbl = coarse_lbl
	    self.fine_lbl = fine_lbl
	    self.question = question

test_global_questions = []
chosen_global_questions = []

def parse_training_file(filename):
    questions = []
    with open(filename) as f:
        lines = f.readlines()
    f.close()

    for line in lines:
        fine_separator = line.find(":")
        question_separator = line.find(" ")

        coarse_lbl = line[:fine_separator]
        fine_lbl = line[:question_separator]
        question = preprocess_question(line[question_separator:].strip())

        questions.append(Question(coarse_lbl, fine_lbl, question))
    
    return questions

def parse_test_file(filename):
    questions = []
    with open(filename) as f:
        lines = f.readlines()
    f.close()

    for line in lines:
        question = preprocess_question(line.strip())
        questions.append(question)
    
    return questions


def preprocess_question(question):
    return stem(remove_stopwords(tokenize(standardize(question))))

def standardize(question):
    #make everything lowercase and remove punctuation and some symbols
    return re.sub("[?|\.|!|:|,|;|`|'|\"]", '', question).lower()

def tokenize(question):
    return word_tokenize(question)

def remove_stopwords(question):
    question_words = set(['what', 'which', 'who', 'why', 'when', 'how', 'where', 'whose'])
    extra_stopwords = ['&', 'first', 'second', 'go', 'one', 'two', 'four','five','%','=']
    stopword_set = set(stopwords.words('english') + extra_stopwords) - question_words

    filtered_question = [w for w in question if not w in stopword_set]

    return filtered_question

def stem(question):
    stemmer = SnowballStemmer("english")
    stemmed_question = [stemmer.stem(w) for w in question ]
    return stemmed_question

def get_label(question, coarseness):
    if(coarseness == "-coarse"):
        return question.coarse_lbl
    elif(coarseness == "-fine"):
        return question.fine_lbl
        
def predict_labels(test_questions, known_questions, coarseness):
    labels = []
    n_known_questions = len(known_questions)
    for test_question in test_questions:
        test_global_questions.append(test_question)
        smallest_dist = 5000
        closest_question = None
        for i in range(n_known_questions):
            dist = jaccard_distance(set(test_question), set(known_questions[i].question))
            if(dist < smallest_dist):
                smallest_dist = dist
                closest_question = known_questions[i]
        if(closest_question != None):
            labels.append(get_label(closest_question, coarseness))
            chosen_global_questions.append(closest_question.question)
        else:
            labels.append(None)

    return labels


def main():
    coarseness = sys.argv[1]
    training_file = sys.argv[2]
    test_file = sys.argv[3]

    known_questions = parse_training_file(training_file)

    test_questions = parse_test_file(test_file)

    #now time to compare and decide stuff

    labels = predict_labels(test_questions, known_questions, coarseness)

    
    with open("treated_known_questions.txt", 'w') as known:
        for known_question in chosen_global_questions:
            for item in known_question:
                known.write(item + " ")
            known.write("\n")
    known.close()

    with open("treated_test_questions.txt", 'w') as test:
        for test_question in test_global_questions:
            for item in test_question:
                test.write(item + " ")
            test.write("\n")
    test.close()


    for label in labels:
        print("{}".format(label))


if __name__ == "__main__":
    main()