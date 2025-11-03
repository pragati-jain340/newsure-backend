from django.urls import path
from . import views

urlpatterns = [
    # path('', views.home),
    path('api/verify/', views.verify_claim),
]