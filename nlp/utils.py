import json
import pickle
import random
import time
from datetime import date

import numpy as np
import pandas as pd
from keras.models import load_model

from nlp import constants
from nlp.data import preprocess

data = pickle.load(open(constants.MODEL_PATH + constants.DATA_NAME, "rb"))
words = data['words']
classes = data['classes']
intents = json.loads(open(constants.DATA + constants.INTENT_FILE).read())

with open(constants.MODEL_PATH + constants.BOT_MODEL, 'rb') as f:
    model = pickle.load(f)

cardio = load_model(constants.MODEL_PATH + constants.CARDIO_MODEL)


def classify_local(sentence):
    # generate probabilities from the model
    input_data = pd.DataFrame([preprocess.bow(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    # filter out predictions below a threshold, and provide intent index
    results = [[i, r] for i, r in enumerate(results) if r > constants.ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    # return tuple of intent and probability
    return return_list


def get_response(intent):
    for i in intents['intents']:
        if intent == i['tag']:
            id = random.randrange(len(i['responses']))
            response = i['responses'][id]
            return response


def input_module():
    print('Bot: What is your day of birth?')
    day = int(input("You: "))
    print('Bot: What is your month of birth?')
    month = int(input("You: "))
    print('Bot: In which year were you born?')
    year = int(input("You: "))
    print('Bot: Are you a boy or a girl?')
    time.sleep(0.5)
    print('Bot: Actually, I prefer numbers, so 0 for female, 1 for male please.')
    gender = input("You: ")
    print('Bot: What is your height?')
    height = float(input("You: "))  # in cm
    print('Bot: What is your weight?')
    weight = float(input("You: "))  # in kilograms
    print('Bot: What is your blood pressure? One line for high, one line for low please.')
    ap_hi = int(input("You: "))  # Systolic blood pressure
    ap_low = int(input("You: "))  # Diastolic blood pressure
    print('Bot: How high is your cholesterol level on a scale of 1 to 3?')
    cholesterol = int(input("You: "))  # 1: normal, 2: above normal, 3: well above normal
    print('Bot: Now do the same with glucose please.')
    gluc = int(input("You: "))  # 1: normal, 2: above normal, 3: well above normal
    print('Bot: Do you smoke? I would appreciate 0 or 1 as an answer.')
    smoke = input("You: ")  # 1 if you smoke, 0 if not
    print('Bot: Do you drink? Again, same as the last one.')
    alco = input("You: ")  # 1 if you drink alcohol, 0 if not
    print('Bot: Final question. Do you exercise? Remember, I can only take in numbers.')
    active = input("You: ")
    return day, month, year, gender, height, weight, ap_hi, ap_low, cholesterol, gluc, smoke, alco, active


def create_test_case(day, month, year, gender, height, weight, ap_hi, ap_low, cholesterol, gluc, smoke, alco, active):
    f_date = date(year, month, day)
    l_date = date.today()
    delta = l_date - f_date
    agedays = delta.days

    df = pd.read_csv(constants.DATA + constants.CARDIO_DATA, sep=';', header=0)

    agedayscale = (agedays - df["age"].min()) / (df["age"].max() - df["age"].min())
    heightscale = (height - df["height"].min()) / (df["height"].max() - df["height"].min())
    weightscale = (weight - df["weight"].min()) / (df["weight"].max() - df["weight"].min())
    sbpscale = (ap_hi - df["ap_hi"].min()) / (df["ap_hi"].max() - df["ap_hi"].min())
    dbpscale = (ap_low - df["ap_lo"].min()) / (df["ap_lo"].max() - df["ap_lo"].min())
    cholesterolscale = (cholesterol - df["cholesterol"].min()) / (df["cholesterol"].max() - df["cholesterol"].min())
    glucscale = (gluc - df["gluc"].min()) / (df["gluc"].max() - df["gluc"].min())

    single = np.array(
        [agedayscale, gender, heightscale, weightscale, sbpscale, dbpscale, cholesterolscale, glucscale, smoke, alco,
         active])
    singledf = pd.DataFrame(single)
    final = singledf.transpose()
    return final


def show_result(final):
    if cardio.predict(final) >= 0.5:
        print('Bot: Chances are you have some kind of cardiovascular disease. Better go get a doctor.')
    elif cardio.predict(final) >= 0.3:
        print('Bot: You are healthy... Probably. But you should still be careful with your habits.')
    else:
        print('Bot: You are perfectly healthy. Keep up the good work!')

