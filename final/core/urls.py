from django.conf import settings
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('video/', views.recognition, name='video'),
    path('upload/', views.select_video, name='upload'),
]
