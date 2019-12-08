import os

from nlp import constants
from nlp.data import preprocess, bot_data
from nlp.models.model import BotModel, CardioModel

if not os.path.exists(constants.MODEL_PATH + constants.BOT_MODEL):
    words, classes, docs = preprocess.process_tfidf(constants.DATA + constants.INTENT_FILE)
    bot = BotModel(constants.BOT_MODEL, words, classes, docs)
    bot.add_training_ops()
    bot.build()

if not os.path.exists(constants.MODEL_PATH + constants.CARDIO_MODEL):
    xdf, ydf = bot_data.process_cardio(constants.DATA + constants.CARDIO_DATA)
    cardio = CardioModel(constants.CARDIO_MODEL, xdf, ydf)
    cardio.add_training_ops()
    cardio.build()
    cardio.evaluate()
