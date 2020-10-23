from nltk import WordNetLemmatizer
import numpy as np
from collections import Counter

def cosine_similarity(a1, a2):
    return np.dot(a1, np.transpose(a2)) / (np.linalg.norm(a1) * np.linalg.norm(a2))

def tf_idf(test_questions, known_questions):
    questions_list = [test_questions, [q.question for q in known_questions]]
    tfs = [[], []]
    doc_terms = [{}, {}]
    terms = set([])

    for qn in range(len(questions_list)):
        questions = questions_list[qn]
        n_questions = len(questions)
        for i in range(n_questions):
            tfs[qn].append({})
            for word in questions[i]:
                terms.add(word)
                if(word in tfs[qn][i]):
                    tfs[qn][i][word] += 1/len(questions[i])
                else:
                    tfs[qn][i][word] = 1/len(questions[i])
                    if(word in doc_terms[qn]):
                        doc_terms[qn][word] += 1
                    else:
                        doc_terms[qn][word] = 1

    terms = list(terms)
    n_terms = len(terms)

    test_tf_idf = np.zeros((len(questions_list[0]), n_terms))
    known_tf_idf = np.zeros((len(questions_list[1]), n_terms))
    tf_idf = [test_tf_idf, known_tf_idf]
    
    for qn in range(len(questions_list)):
        n_questions = len(questions_list[qn])
        for i in range(n_questions):
            for j in range(len(terms)):
                if(terms[j] in tfs[qn][i]):
                    df = doc_terms[qn][terms[j]]
                    tf_idf[qn][i][j] = tfs[qn][i][terms[j]] * np.log(n_questions/df + 1)

    return tf_idf[0], tf_idf[1]

def predict_labels_tf_idf(test_questions, known_questions, coarseness):
    test_tf_idf, known_tf_idf = tf_idf(test_questions, known_questions)

    labels = []
    n_known_questions = len(known_questions)

    for i in range(len(test_questions)):
        best_similarity = 0
        closest_question = None
        for j in range(n_known_questions):
            similarity = cosine_similarity(test_tf_idf[i], known_tf_idf[j])
            if(similarity > best_similarity):
                 best_similarity = similarity
                 closest_question = known_questions[j]
        if(closest_question != None):
            labels.append(get_label(closest_question, coarseness))
        else:
             labels.append(None)

    return labels

def lemma(question):
    lemma = WordNetLemmatizer()
    lemma_verbs = [lemma.lemmatize(w,pos = "v") for w in question]
    lemma_nouns = [lemma.lemmatize(w,pos = "n") for w in lemma_verbs]
    #lemmatized = [wnl().lemmatize(w.lower()) for w in lemma_nouns]
    return lemma_nouns

def get_most_common_words(questions):
    tokens = []
    
    for question in questions:
        for token in question.question:
            tokens.append(token)

    word_count = Counter(tokens)
    word_count.most_common(20)