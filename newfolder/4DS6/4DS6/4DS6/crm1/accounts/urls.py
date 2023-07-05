from django.urls import path

from django.contrib.auth import views as auth_views

from . import views



urlpatterns = [
	path('register/', views.registerPage, name="register"),
	path('login/', views.loginPage, name="login"),  
	path('logout/', views.logoutUser, name="logout"),
    path('upload/', views.upload_file, name='upload_file'),
    path('dashboard/', views.dashboard_view, name='dashboard'),

    path('', views.home, name="home"),
    path('user/', views.userPage, name="user-page"),
    path('success/url/', views.success_view, name='success'),
    path('account/', views.accountSettings, name="account"),
    path('index/', views.indexPage, name="index"),
    path('google_maps/', views.mapsPage, name="google_maps"),
    path('basic_table/', views.tablePage, name="basic_table"),
    path('buttons/', views.buttonPage, name="buttons"),
    path('inbox/', views.inboxPage, name="inbox"),
    path('morris/', views.morrisPage, name="morris"),
    path('Chartjs/', views.ChartjsPage, name="Chartjs"),
    path('flot_chart/', views.flot_chartPage, name="flot_chart"),
    path('xchart/', views.xchartPage, name="xchart"),


    path('reset_password/',
     auth_views.PasswordResetView.as_view(template_name="accounts/password_reset.html"),
     name="reset_password"),

    path('reset_password_sent/', 
        auth_views.PasswordResetDoneView.as_view(template_name="accounts/password_reset_sent.html"), 
        name="password_reset_done"),

    path('reset/<uidb64>/<token>/',
     auth_views.PasswordResetConfirmView.as_view(template_name="accounts/password_reset_form.html"), 
     name="password_reset_confirm"),

    path('reset_password_complete/', 
        auth_views.PasswordResetCompleteView.as_view(template_name="accounts/password_reset_done.html"), 
        name="password_reset_complete"),



]

