from django.conf.urls import url
from .views import test, webhook

urlpatterns = [
    url(r'^test$',test, name='test'),
    url(r'^webhook$',webhook, name='webhook'),
]