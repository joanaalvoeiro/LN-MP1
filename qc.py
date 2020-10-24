import re
import sys
import string
import numpy as np

from nltk import bigrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.metrics.distance import jaccard_distance

class Question:
    def __init__(self, coarse_lbl, fine_lbl, question):
	    self.coarse_lbl = coarse_lbl
	    self.fine_lbl = fine_lbl
	    self.question = question


#----------------------------------------------------------------------------------
#           parsing functions
#----------------------------------------------------------------------------------

def parse_training_file(filename,coarseness):
    questions = []
    tokens = []
    with open(filename) as f:
        lines = f.readlines()
    f.close()

    for line in lines:
        fine_separator = line.find(":")
        question_separator = line.find(" ")

        coarse_lbl = line[:fine_separator]
        fine_lbl = line[:question_separator]
        question = preprocess_question(line[question_separator:].strip(),coarseness)

        if coarseness == "-fine":
            question = layout_form_fine(question)

        questions.append(Question(coarse_lbl, fine_lbl, question))


    return questions


def parse_test_file(filename,coarseness):
    questions = []
    with open(filename) as f:
        lines = f.readlines()
    f.close()

    for line in lines:
        question = preprocess_question(line.strip(),coarseness)

        if coarseness == "-fine":
            question = layout_form_fine(question)

        questions.append(question)
    
    return questions

def layout_form_fine(question):
    q = []
    for pair in question:
        for w in pair:
            q += [w]
    return q

#----------------------------------------------------------------------------------
#           preprocessing
#----------------------------------------------------------------------------------

def preprocess_question(question,coarseness):
    if coarseness == "-fine":
        return bigramify(remove_stopwords(tokenize(standardize(question)),coarseness))
    else:
        return stem(remove_stopwords(tokenize(standardize(question)),coarseness))


def standardize(question):
    #remove punctuation and some symbols, lowercase
    standardized = "".join([char for char in question if char not in string.punctuation])
    return standardized.lower()


def tokenize(question):
    return word_tokenize(question)


def bigramify(question):
    return list(bigrams(question))


def remove_stopwords(question,coarseness):
    question_words =  set(['what', 'which', 'who', 'why', 'when', 'how', 'where', 'whose'])

    if(coarseness == '-fine'):
        extra_stopwords = ['first' ,'one', 'four']
    else:
        extra_stopwords = ['world', 'go','one', 'four']

    stopword_set = set(stopwords.words('english') + extra_stopwords) - question_words
    filtered_question = [w for w in question if not w in stopword_set]

    return filtered_question


def stem(question):
    stemmer = SnowballStemmer("english")
    
    return [stemmer.stem(w) for w in question]


#----------------------------------------------------------------------------------
#           label prediction
#----------------------------------------------------------------------------------

def get_label(question, coarseness):
    if(coarseness == "-coarse"):
        return question.coarse_lbl
    elif(coarseness == "-fine"):
        return question.fine_lbl

def predict_labels(test_questions, known_questions, coarseness):
    labels = []
    n_known_questions = len(known_questions)
    for test_question in test_questions:
        smallest_dist = 5000
        closest_question = None
        for i in range(n_known_questions):
            dist = jaccard_distance(set(test_question), set(known_questions[i].question))
            if(dist < smallest_dist):
                 smallest_dist = dist
                 closest_question = known_questions[i]
        if(closest_question != None):
            labels.append(get_label(closest_question, coarseness))
        else:
             labels.append(None)
    return labels

#----------------------------------------------------------------------------------
#           main
#----------------------------------------------------------------------------------  

def main():
    coarseness = sys.argv[1]
    training_file = sys.argv[2]
    test_file = sys.argv[3]

    known_questions = parse_training_file(training_file,coarseness)

    test_questions = parse_test_file(test_file,coarseness)

    labels = predict_labels(test_questions, known_questions, coarseness)

    for label in labels:
        print("{}".format(label))


if __name__ == "__main__":
    main()