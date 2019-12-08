import json
import pickle
import random
import time
from datetime import date

import numpy as np
import pandas as pd

from nlp import constants
from nlp.data import preprocess
from nlp.data_utils import MyIOError, MyTypeError


# from keras.models import load_model


def load_prediction_model(filename):
    try:
        model = pickle.load(open(filename, "rb"))
    except IOError:
        raise MyIOError(filename)
    return model


data = load_prediction_model(constants.MODEL_PATH + constants.DATA_NAME)
words = data['words']
classes = data['classes']
docs = data['documents']

context = {}

intents = json.loads(open(constants.DATA + constants.INTENT_FILE, encoding="utf8").read())


def classify_local(sentence, model):
    # generate probabilities from the model
    # input_data = pd.DataFrame([preprocess.bow(sentence, words)], dtype=float, index=['input'])
    input_data = pd.DataFrame([preprocess.tfidf(sentence, words, docs)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    # print(model.predict([input_data]))
    # print(results)
    # filter out predictions below a threshold, and provide intent index
    results = [[i, r] for i, r in enumerate(results) if r > constants.ERROR_THRESHOLD]
    # sort by strength of probability
    # print(results)
    results.sort(key=lambda x: x[1], reverse=True)
    print(results)
    return_list = []
    for r in results:
        # print(r)
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    # return tuple of intent and probability
    return return_list


def get_response(message, model, user_id='123', show_details=False):
    # intent = classify_local(message, model)[0]['intent']

    result = classify_local(message, model)
    # print(result)
    intent = result[0]['intent']

    for i in intents['intents']:
        if intent == i['tag']:
            if 'context' in i:
                if show_details:
                    print('context:', i['context'])
                    if i['context'] != '':
                        context[user_id] = i['context']
            if not 'link' in i or \
                    (user_id in context and 'link' in i and i['link'] == context[user_id]):
                if show_details:
                    print('tag:', i['tag'])
                id = random.randrange(len(i['responses']))
                response = i['responses'][id]
                # print(response)
                return response


def input_module():
    try:
        time.sleep(0.5)
        print('Bot: What is your birthday?')
        time.sleep(0.5)
        print('Bot: I am not American, unfortunately. So I only take the dd/mm/yyyy format')
        full_date = input("You: ")
        day, month, year = full_date.split('/')
        f_date = date(int(year), int(month), int(day))
        print('Bot: Are you a boy or a girl?')
        time.sleep(0.5)
        print('Bot: Actually, I prefer numbers, so 1 for female, 2 for male please.')
        gender = input("You: ")
        print('Bot: What is your height in centimeters?')
        height = float(input("You: "))  # in cm
        print('Bot: What is your weight in kilograms?')
        weight = float(input("You: "))  # in kilograms
        print('Bot: What is your blood pressure? One line for high, one line for low please.')
        ap_hi = int(input("You: "))  # Systolic blood pressure
        ap_low = int(input("You: "))  # Diastolic blood pressure
        print('Bot: How high is your cholesterol level on a scale of 1 to 3?')
        cholesterol = int(input("You: "))  # 1: normal, 2: above normal, 3: well above normal
        print('Bot: Now do the same with your glucose level glucose please.')
        gluc = int(input("You: "))  # 1: normal, 2: above normal, 3: well above normal
        print('Bot: Do you smoke? I would appreciate 0 or 1 as an answer.')
        smoke = int(input("You: "))  # 1 if you smoke, 0 if not
        print('Bot: Do you drink? Again, same as the last one.')
        alco = int(input("You: "))  # 1 if you drink alcohol, 0 if not
        print('Bot: Final question with this format. Do you exercise?')
        active = int(input("You: "))
    except Exception:
        raise MyTypeError()
    return f_date, gender, height, weight, ap_hi, ap_low, cholesterol, gluc, smoke, alco, active


def create_test_case(f_date, gender, height, weight, ap_hi, ap_low, cholesterol, gluc, smoke, alco, active):
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


def show_result(cardio, final):
    if cardio.predict(final) >= 0.5:
        print('Bot: Chances are you have some kind of cardiovascular disease. Better go get a doctor.')
    elif cardio.predict(final) >= 0.3:
        print('Bot: You are healthy... Probably. But you should still be careful with your habits.')
    else:
        print('Bot: You are perfectly healthy. Keep up the good work!')




