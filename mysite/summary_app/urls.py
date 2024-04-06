from django.urls import path

from . import views

urlpatterns = [
    path('', views.summarize_page, name='summary_page'),
    path("content/", views.summarize_content, name='summary_content')
]
