from django.contrib import admin
from django.urls import path
from ResNet.views import ImagePredictionAPIView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/predict/', ImagePredictionAPIView.as_view(), name='image-prediction'),
]
