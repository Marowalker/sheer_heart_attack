import random
from _collections import defaultdict

import nlp.constants as constants
from nlp.utils import load_prediction_model, classify_local, get_response, create_test_case, show_result, input_module

cardio = load_prediction_model(constants.MODEL_PATH + constants.CARDIO_MODEL)

model = load_prediction_model(constants.MODEL_PATH + constants.BOT_MODEL)


def test_dialog():
    sent = ''
    intent = ''
    while intent != 'goodbye':
        sent = input("You: ")
        # print(classify_local(sent, model)[0]['intent'])
        intent = classify_local(sent, model)[0]['intent']
        print("Bot: " + str(get_response(sent, model)))
        if intent == 'predict_disease':
            print('Changing to prediction module...')
            day, gender, height, weight, ap_hi, ap_low, cholesterol, gluc, smoke, alco, active = input_module()
            test_case = create_test_case(day, gender, height, weight, ap_hi, ap_low, cholesterol, gluc, smoke, alco,
                                         active)
            show_result(cardio, test_case)


def test_eval():
    file = open('data/test.txt', 'r')
    eval_dict = defaultdict()
    lines = file.readlines()
    random.shuffle(lines)
    for line in lines:
        s, i = line.rsplit('|', 1)
        i = i.strip()
        eval_dict[i] = s

    count = 0
    for id in eval_dict:
        # print(id)
        it = classify_local(eval_dict[id], model)[0]['intent']
        # print(it)
        if it == id:
            count += 1

    print(count)
    print(len(eval_dict))
    print('model accuracy: ', float(count / len(eval_dict)))


# test_dialog()
test_eval()
