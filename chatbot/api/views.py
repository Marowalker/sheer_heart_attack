from rest_framework.decorators import api_view
from django.http import JsonResponse
from rest_framework.response import Response
from chatbot import settings
import requests
import json
# from chatbot.core.nlp.utils import get_response

@api_view(['GET'])
def test(request):
    # print(get_response('hi'))
    return JsonResponse({
        'data': 'test'
    }, status=200)

@api_view(['GET','POST'])
def webhook(request):
    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        message=body['entry'][0]['messaging'][0]['message']['text']
        sender=body['entry'][0]['messaging'][0]['sender']['id']
        answer_message(sender,message)
        return Response('EVENT_RECEIVED', status=200)
    if request.method == 'GET':
        mode=request.GET.get('hub.mode', None)
        verify_token=request.GET.get('hub.verify_token', None)
        challenge=request.GET.get('hub.challenge', None)
        if not settings.VALIDATION_TOKEN==verify_token:
            return Response('FAIL', status=403)
        print('verify_token success')
        return Response(int(challenge), status=200)
    return Response('FAIL', status=403)


def answer_message(fb_user_id, message):
    url = settings.SERVER_URL
    message_data = {
        "recipient": {
            "id": fb_user_id
        },
        "message": {
            "text": message+" gì thế bạn?"
        }
    }
    response = requests.post(
        url,
        params={"access_token": settings.PAGE_ACCESS_TOKEN},
        json=message_data)
    return response, message_data