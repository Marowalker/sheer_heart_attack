from rest_framework.decorators import api_view
from django.http import JsonResponse
from chatbot.core.nlp.utils import get_response

@api_view(['GET'])
def test(request):

    return JsonResponse({
        'data': 'get_response'
    }, status=200)