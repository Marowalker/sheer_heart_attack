import argparse

from nltk.stem import WordNetLemmatizer

parser = argparse.ArgumentParser(description='Simple MLP for chatbot')
parser.add_argument('-i', help='Job identity', type=int, default=0)
parser.add_argument('-rb', help='Rebuild data', type=int, default=0)
parser.add_argument('-e', help='Number of epochs', type=int, default=50)

opt = parser.parse_args()
print('Running opt: {}'.format(opt))

JOB_IDENTITY = opt.i
IS_REBUILD = opt.rb
EPOCHS = opt.e

LEMMA = WordNetLemmatizer()

NLP = 'nlp/'
MODEL_PATH = 'models/'
DATA = 'data/'

INTENT_FILE = 'intents.json'
CARDIO_DATA = 'cardio_train.csv'
IGNORE_WORDS = ['?']
ERROR_THRESHOLD = 0.25

BOT_MODEL = 'sheer_heart_attack.pkl'
DATA_NAME = 'sheer_heart_attack_data.pkl'
CARDIO_MODEL = 'bite_the_dust.h5'
