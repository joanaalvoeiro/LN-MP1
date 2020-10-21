import sys
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.metrics.distance import jaccard_distance
from nltk import bigrams
from collections import Counter
from nltk import WordNetLemmatizer



class Question:
    def __init__(self, coarse_lbl, fine_lbl, question):
	    self.coarse_lbl = coarse_lbl
	    self.fine_lbl = fine_lbl
	    self.question = question

test_global_questions = []
chosen_global_questions = []


def parse_training_file(filename,coarseness):
    questions = []
    tokens = []
    question_words = ['what', 'which', 'who', 'why', 'when', 'how', 'where', 'whose']
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

    for question in questions:
        for token in question.question:
            if token not in question_words:
                tokens.append(token)

    word_count = Counter(tokens)
    most_common = word_count.most_common(20)
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


def preprocess_question(question,coarseness):
    if coarseness == "-fine":
        return bigrams_aux(remove_stopwords(tokenize(standardize(question)),coarseness))
    else:
        return stem(remove_stopwords(tokenize(standardize(question)),coarseness))


def standardize(question):
    #make everything lowercase and remove punctuation and some symbols
    return re.sub("[?|\.|!|:|,|;|`|'|\"]", '', question)


def tokenize(question):
    return word_tokenize(question)


def bigrams_aux(question):
    return list(bigrams(question))


def remove_stopwords(question,coarseness):
    question_words = set(['what', 'which', 'who', 'why', 'when', 'how', 'where', 'whose'])

    if(coarseness == '-fine'):
        extra_stopwords = ['&', 'first','one','four','five','fourth']
    else:
        extra_stopwords = ['&','name','world','first','second','go','one', 'two', 'four','five','get','origin']


    stopword_set = set(stopwords.words('english')+ extra_stopwords) - question_words
    filtered_question = [w for w in question if not w in stopword_set]
    return filtered_question


def stem(question):
    lemma = WordNetLemmatizer()
    lemma_verbs = [lemma.lemmatize(w,pos = "v") for w in question]
    lemma_nouns = [lemma.lemmatize(w,pos = "n") for w in lemma_verbs]
    return lemma_nouns


def get_label(question, coarseness):
    if(coarseness == "-coarse"):
        return question.coarse_lbl
    elif(coarseness == "-fine"):
        return question.fine_lbl


def layout_form_fine(question):
    q = []
    for pair in question:
        for w in pair:
            q += [w]
    return q


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

    known_questions = parse_training_file(training_file,coarseness)

    test_questions = parse_test_file(test_file,coarseness)

    #now time to compare and decide stuff

    labels = predict_labels(test_questions, known_questions, coarseness)

    
    with open("treated_known_questions.txt", 'w') as known:
        for known_question in chosen_global_questions:
            for item in known_question:
                known.write(str(item) + " ")
            known.write("\n")
    known.close()

    with open("treated_test_questions.txt", 'w') as test:
        for test_question in test_global_questions:
            for item in test_question:
                test.write(str(item) + " ")
            test.write("\n")
    test.close()


    for label in labels:
        print("{}".format(label))


if __name__ == "__main__":
    main()