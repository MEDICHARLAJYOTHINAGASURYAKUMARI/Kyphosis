from django.shortcuts import render,HttpResponse
from django.contrib import messages
from users.models import UserRegistrationModel

# Create your views here.

def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')
        elif usrid == 'Admin' and pswd == 'Admin':
            return render(request, 'admins/AdminHome.html')
        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})


def AdminHome(request):
    return render(request, 'admins/AdminHome.html')


def ViewRegisteredUsers(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/RegisteredUsers.html', {'data': data})


def AdminActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request, 'admins/RegisteredUsers.html', {'data': data})
    

# views.py

from django.shortcuts import render, redirect

def AdminDeleteUser(request):
    if request.method == 'GET':
        user_id = request.GET.get('uid')
        if user_id:
            UserRegistrationModel.objects.filter(id=user_id).delete()
        data = UserRegistrationModel.objects.all()
        return render(request,'admins/RegisteredUsers.html',{'data': data})  # Redirect to the list view (update the URL name if different)
