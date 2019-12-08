import json
import math

import nltk
import numpy as np

from nlp import constants


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [constants.LEMMA.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def tf(word, word_list):
    count = 0
    for token in word_list:
        if word == token:
            count += 1
    return float(count/len(word_list))


def idf(word, documents):
    docs = [d[0] for d in documents]
    N = len(docs)
    count = 0
    for d in docs:
        if d.find(word) != -1:
            count += 1
    count += 1
    N += 1
    return math.log(N/float(count))


def tfidf_score(word, sentence, documents):
    sentence_words = clean_up_sentence(sentence)
    _tf = tf(word, sentence_words)
    _idf = idf(word, documents)
    score = _tf * _idf
    return score


def tfidf(sentence, words, documents):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0.0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = tfidf_score(s, sentence, documents)
    return np.array(bag)


def process(intent_file):
    words = []
    classes = []
    documents = []
    intents = json.loads(open(intent_file, 'r', encoding='utf-8').read())
    # loop through each sentence in our intents patterns
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # tokenize each word in the sentence
            w = nltk.word_tokenize(pattern)
            # add to our words list
            words.extend(w)
            # add to documents in our corpus
            documents.append((w, intent['tag']))
            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    # stem and lower each word and remove duplicates
    words = [constants.LEMMA.lemmatize(w.lower()) for w in words if w not in constants.IGNORE_WORDS]
    words = sorted(list(set(words)))
    # sort classes
    classes = sorted(list(set(classes)))
    return words, classes, documents


def process_tfidf(intent_file):
    words = []
    classes = []
    documents = []
    intents = json.loads(open(intent_file, 'r', encoding='utf-8').read())
    # loop through each sentence in our intents patterns
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # tokenize each word in the sentence
            w = nltk.word_tokenize(pattern)
            # add to our words list
            words.extend(w)
            # add to documents in our corpus
            documents.append((pattern, intent['tag']))
            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    # stem and lower each word and remove duplicates
    words = [constants.LEMMA.lemmatize(w.lower()) for w in words if w not in constants.IGNORE_WORDS]
    words = sorted(list(set(words)))
    # sort classes
    classes = sorted(list(set(classes)))
    return words, classes, documents


def test_len(intent_file):
    intents = json.loads(open(intent_file, 'r', encoding='utf-8').read())
    # loop through each sentence in our intents patterns
    for intent in intents['intents']:
        print(intent['tag'], len(intent['patterns']))



