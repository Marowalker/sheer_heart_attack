import random

import nlp.constants as constants
from nlp.utils import load_prediction_model, classify_local, get_response, create_test_case, show_result, input_module

cardio = load_prediction_model(constants.MODEL_PATH + constants.CARDIO_MODEL)

model = load_prediction_model(constants.MODEL_PATH + constants.BOT_MODEL)


def test_dialog():
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
    file = open('data/quest.txt', 'r')
    # eval_dict = defaultdict()
    lines = file.readlines()
    random.shuffle(lines)
    temp = []
    for line in lines:
        s, i = line.rsplit('|', 1)
        i = i.strip()
        temp.append(tuple([s, i]))
        # eval_dict[i] = s

    count = 0
    for t in temp:
        # print(id)
        print(t[1])
        it = classify_local(t[0], model)[0]['intent']
        print(it)
        if it == t[1]:
            count += 1

    print(count)
    print(len(lines))
    print('model accuracy: ', float(count / len(lines)))


# test_dialog()
test_eval()
