import random

import numpy as np
import pandas as pd
from sklearn import preprocessing

from chatbot.core.nlp import constants


def get_training_data(words, classes, documents):
    training = []
    # create an empty array for our output
    output_empty = [0] * len(classes)
    # training set, bag of words for each sentence
    for doc in documents:
        # initialize our bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # stem each word - create base word, in attempt to represent related words
        pattern_words = [constants.LEMMA.lemmatize(word.lower()) for word in pattern_words]
        # create our bag of words array with 1, if word match found in current pattern
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        # output is a '0' for each tag and '1' for current tag (for each pattern)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])
    # shuffle our features and turn into np.array
    random.shuffle(training)
    training = np.array(training)
    # create train and test lists. X - patterns, Y - intents
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    return train_x, train_y


def process_cardio(filename):
    # data = pd.read_csv(filename)
    df = pd.read_csv(filename, sep=';', header=0)
    dfcol = df.columns
    # print(df.head())

    scaler = preprocessing.MinMaxScaler()
    dfscale = scaler.fit_transform(df)
    dfscale2 = pd.DataFrame(dfscale, columns=dfcol)
    xdf = dfscale2.iloc[:, 0:11]
    ydf = dfscale2.iloc[:, -1]
    return xdf, ydf
