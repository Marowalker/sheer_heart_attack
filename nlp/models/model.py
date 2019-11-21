import pickle

import numpy as np
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

import nlp.data.bot_data as train
from nlp import constants


class BotModel:
    def __init__(self, model_name, words, classes, docs):
        self.model_name = model_name
        self.words = words
        self.classes = classes
        self.docs = docs
        self.model = Sequential()
        self.sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.train_x, self.train_y = train.get_training_data(words, classes, docs)

    def add_training_ops(self):
        self.model.add(Dense(128, input_shape=(len(self.train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(self.train_y[0]), activation='softmax'))

    def build(self):
        self.model.compile(loss='categorical_crossentropy', optimizer=self.sgd, metrics=['accuracy'])
        # Fit the model
        self.model.fit(np.array(self.train_x), np.array(self.train_y), epochs=200, batch_size=5, verbose=1)

        # save model to file
        pickle.dump(self.model, open(constants.MODEL_PATH + self.model_name, "wb"))

        # save all of our data structures
        pickle.dump({'words': self.words, 'classes': self.classes, 'train_x': self.train_x, 'train_y': self.train_y},
                    open(constants.MODEL_PATH + constants.DATA_NAME, "wb"))


class CardioModel:
    def __init__(self, model_name, xdf, ydf):
        self.model = Sequential()
        self.model_name = model_name
        self.train_x, self.test_x, self.train_y, self.test_y = \
            train_test_split(xdf, ydf, test_size=0.2, random_state=123, stratify=ydf)

    def add_training_ops(self):
        self.model.add(Dense(25, input_dim=11, activation='softsign', kernel_constraint=maxnorm(2)))
        # model.add(Dropout(0))
        self.model.add(Dense(5, activation='softsign'))
        # model.add(Dropout(0))
        self.model.add(Dense(3, activation='softsign'))
        # model.add(Dropout(0))
        self.model.add(Dense(1, activation='sigmoid'))

    def build(self):
        self.model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])

        self.model.fit(self.train_x, self.train_y, epochs=50, batch_size=50, verbose=0)

        # self.model.save(constants.MODEL_PATH + self.model_name)

        pickle.dump(self.model, open(constants.MODEL_PATH + self.model_name, "wb"))

    def evaluate(self):
        score = self.model.evaluate(self.train_x, self.train_y)
        print("\n Training Accuracy:", score[1])
        score = self.model.evaluate(self.test_x, self.test_y)
        print("\n Testing Accuracy:", score[1])

