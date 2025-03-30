"""ElectricityForecasting URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from KyphosisDiseaseClassification import views as mainView
from admins import views as admins
from users import views as usr

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", mainView.index, name='index'),
    path("index/", mainView.index, name="index"),
    path("logout/", mainView.logout, name="logout"),
    path("UserLogin/", mainView.UserLogin, name="UserLogin"),
    path("AdminLogin/", mainView.AdminLogin, name="AdminLogin"),
    path("UserRegister/", mainView.UserRegister, name="UserRegister"),

    # User Side Views
    path("UserRegisterActions/", usr.UserRegisterActions, name="UserRegisterActions"),
    path("UserLoginCheck/", usr.UserLoginCheck, name="UserLoginCheck"),
    path("UserHome/", usr.UserHome, name="UserHome"),
    path("DatasetView/", usr.DatasetView, name="DatasetView"),
    path("ml_training/", usr.ml_training, name="ml_training"),
    path("predict_kyphosis/", usr.predict_kyphosis, name="predict_kyphosis"),
    path("ada/",usr.ada,name='ada'),
    path("xg/",usr.xg,name='xg'),
    path("ml_training1/", usr.ml_training1, name="ml_training1"),
    path("predict_kyphosis1/", usr.predict_kyphosis1, name="predict_kyphosis1"),
    path("gradent/",usr.gradent,name='gradent'),
    path("ml_training2/", usr.ml_training2, name="ml_training2"),
    path("ml_training3/", usr.ml_training3, name="ml_training3"),
    path("predict_kyphosis2/", usr.predict_kyphosis2, name="predict_kyphosis2"),  
    path("predict_kyphosis3/", usr.predict_kyphosis3, name="predict_kyphosis3"),
    path("compare_models/", usr.compare_models, name="compare_models"),
    
    # Admin Side Views
    path("AdminLoginCheck/", admins.AdminLoginCheck, name="AdminLoginCheck"),
    path("AdminHome/", admins.AdminHome, name="AdminHome"),
    path("ViewRegisteredUsers/", admins.ViewRegisteredUsers, name="ViewRegisteredUsers"),
    path("AdminActivaUsers/", admins.AdminActivaUsers, name="AdminActivaUsers"),
    path("AdminDeleteUser/",admins.AdminDeleteUser,name="AdminDeleteUser")
]
