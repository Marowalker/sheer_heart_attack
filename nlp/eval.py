from nlp.utils import classify_local, get_response, input_module, create_test_case, show_result


sent = ''
intent = ''
while intent != 'goodbye':
    sent = input("You: ")
    # print(utils.classify_local(sent)[0]['intent'])
    intent = classify_local(sent)[0]['intent']
    print('Bot: ' + get_response(sent))
    if intent == 'predict_disease':
        print('Changing to prediction module...')
        day, gender, height, weight, ap_hi, ap_low, cholesterol, gluc, smoke, alco, active = input_module()
        test_case = create_test_case(day, gender, height, weight, ap_hi, ap_low, cholesterol, gluc, smoke, alco, active)
        show_result(test_case)


