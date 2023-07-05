from django.shortcuts import render, redirect 
from django.http import HttpResponse
from django.forms import inlineformset_factory
from django.contrib.auth.forms import UserCreationForm
import os
import configparser
from PIL import Image
from django.http import HttpResponseRedirect
from django.shortcuts import render
import tempfile
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import Group
# Create your views here.
from .models import *
from .forms import OrderForm, CreateUserForm, CustomerForm, UploadFileForm
from .filters import OrderFilter
from .decorators import unauthenticated_user, allowed_users, admin_only
from pdf2image import convert_from_path
from .ocr.Cv import CVAnalyzer  # Importez la classe CVAnalyzer depuis votre fichier cvanalyzer.py
from ultralytics import YOLO
from .ocr.Matching import JobMatching
from .ocr.Skills import technical_skills_list
from .ocr.Profile import profil_list
@unauthenticated_user
def registerPage(request):

	form = CreateUserForm()
	if request.method == 'POST':
		form = CreateUserForm(request.POST)
		if form.is_valid():
			user = form.save()
			username = form.cleaned_data.get('username')


			messages.success(request, 'Account was created for ' + username)

			return redirect('login')
		

	context = {'form':form}
	return render(request, 'accounts/register.html', context)


def upload_file(request):
	if request.method == 'POST' and request.FILES['pdf_file']:
		pdf_file = request.FILES['pdf_file']
		destination_path = 'D:/pfe/MesCvs/' + pdf_file.name
		with open(destination_path, 'wb+') as destination:
			for chunk in pdf_file.chunks():
				destination.write(chunk)

		# Convert the PDF to images
		images = convert_from_path(destination_path, dpi=300)
		output_folder = 'D:/pfe/MesCvs/Images/'  # Define the output folder
		os.makedirs(output_folder, exist_ok=True)  # Create the directory if it doesn't exist

		for i, image in enumerate(images):
			# Redimensionner l'image à une taille similaire à celle sur ilovepdf.com (1654x2338 pixels)
			resized_image = image.resize((1654, 2338), Image.ANTIALIAS)

			# Compresser l'image pour réduire la taille du fichier
			image_name = f"image_{i + 1}.jpg"
			image_path = os.path.join(output_folder, image_name)
			resized_image.save(image_path, "JPEG", quality=80, optimize=True)

			# Initialize CVAnalyzer and process the image
			analyzer = CVAnalyzer()
			config = configparser.ConfigParser()
			config.read('config.ini')
			model_path = config['MODEL']['Path']
			analyzer.load_yolo_model(model_path)
			yolo = YOLO()

			try:
				results = analyzer.get_predictions(image_path, confidence=0.8)
				bboxes = results[0].boxes.xyxy  # les coordonnées des boîtes englobantes
				probs = results[0].boxes.conf  # les confiances des prédictions
				names = results[0].boxes.cls

				# Extraction des valeurs de l'image et création de la DataFrame
				df = analyzer.extract_values_from_image(image_path,
														'C:/Users/ASUS/Desktop/Nouveau dossier/newfolder/4DS6/4DS6/4DS6/crm1/accounts/ocr/output/content/output',
														image_name, bboxes, probs, names)
				df = analyzer.clean_and_sort_dataframe('accounts/ocr/my_dataframe.csv')
				df = analyzer.add_technical_skills(df)
				language_keywords = ['francais', 'anglais', 'allemand', 'espagnol', 'italien', 'arab']
				analyzer.extract_languages_from_dataframe(df, language_keywords)
				df = analyzer.extract_contact_info(df)
				df = analyzer.apply_extraction(df)
				new_df = analyzer.replace(df, 'accounts/ocr/Finaloutput.csv')

				# Sauvegarde de la DataFrame modifiée dans la base de données
				try:
					analyzer.save_in_database(new_df)
					print("Data saved to the database successfully.")
				except pyodbc.Error:
					print("Failed to save data to the database. Please check the database connection.")
				job_matching=JobMatching('accounts/ocr/processed_data.csv', 'accounts/ocr/Finaloutput.csv', 'config.ini')
				job_matching.load_data()
				job_matching.initialize_model()
				resultats_matching = job_matching.process_matching()
				print(resultats_matching)
				job_matching.Save_data()
				job_matching.close_connection()
				print("Job matching process completed.")
			except FileNotFoundError:
				print("Image not found. Please provide a valid image path.")
			except Exception as e:
				print("An error occurred:", str(e))
  		# Récupérer les 3 meilleurs emplois
		return render(request, 'success.html', {'success': True,  'resultats_matching': resultats_matching})
	return render(request, 'upload.html', {'success': False})



def dashboard_view(request):
    powerbi_url = "https://app.powerbi.com/reportEmbed?reportId=b28dc991-b808-4eee-a9cc-0569772218dc&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730"
    return render(request, 'dashboard.html', {'powerbi_url': powerbi_url})

@unauthenticated_user
def loginPage(request):

	if request.method == 'POST':
		username = request.POST.get('username')
		password =request.POST.get('password')

		user = authenticate(request, username=username, password=password)

		if user is not None:
			login(request, user)
			return redirect('home')
		else:
			messages.info(request, 'Username OR password is incorrect')

	context = {}
	return render(request, 'accounts/login.html', context)

def logoutUser(request):
	logout(request)
	return redirect('login')

@login_required(login_url='login')
@admin_only
def home(request):
	orders = Order.objects.all()
	customers = Customer.objects.all()

	total_customers = customers.count()

	total_orders = orders.count()
	delivered = orders.filter(status='Delivered').count()
	pending = orders.filter(status='Pending').count()

	context = {'orders':orders, 'customers':customers,
	'total_orders':total_orders,'delivered':delivered,
	'pending':pending }

	return render(request, 'accounts/dashboard.html', context)

def userPage(request):
	return render(request, 'index.html')
def mapsPage(request):
	return render(request, 'google_maps.html')
def indexPage(request):
	return render(request, 'index.html')

def inboxPage(request):
	return render(request, 'inbox.html')
def buttonPage(request):
	return render(request, 'buttons.html')
def tablePage(request):
	return render(request, 'basic_table.html')
def morrisPage(request):
	return render(request, 'morris.html')
def ChartjsPage(request):
	return render(request, 'Chartjs.html')
def flot_chartPage(request):
	return render(request, 'flot_chart.html')
def xchartPage(request):
	return render(request, 'xchart.html')
def success_view(request):
    return render(request, 'success.html')
@login_required(login_url='login')
@allowed_users(allowed_roles=['customer'])
def accountSettings(request):
	customer = request.user.customer
	form = CustomerForm(instance=customer)

	if request.method == 'POST':
		form = CustomerForm(request.POST, request.FILES,instance=customer)
		if form.is_valid():
			form.save()


	context = {'form':form}
	return render(request, 'accounts/account_settings.html', context)


