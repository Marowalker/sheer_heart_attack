import nlp.constants as constants
from nlp.utils import load_prediction_model, classify_local, get_response

sent = ''
intent = ''

cardio = load_prediction_model(constants.MODEL_PATH + constants.CARDIO_MODEL)

model = load_prediction_model(constants.MODEL_PATH + constants.BOT_MODEL)

while intent != 'goodbye':
    sent = input("You: ")
    # print(classify_local(sent, model)[0]['intent'])
    intent = classify_local(sent, model)[0]['intent']
    print("Bot: " + get_response(sent, model))
    # if intent == 'predict_disease':
        # print('Changing to prediction module...')
        # day, gender, height, weight, ap_hi, ap_low, cholesterol, gluc, smoke, alco, active = input_module()
        # test_case = create_test_case(day, gender, height, weight, ap_hi, ap_low, cholesterol, gluc, smoke, alco, active)
        # show_result(cardio, test_case)


