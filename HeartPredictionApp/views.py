from django.shortcuts import render
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'heart_model.pkl')
model = joblib.load(model_path)

def home(request):
    result = None
    if request.method == "POST":
        age = int(request.POST['age'])
        trestbps = int(request.POST['trestbps'])
        chol = int(request.POST['chol'])
        thalach = int(request.POST['thalach'])
        oldpeak = float(request.POST['oldpeak'])

        sex = 1
        cp  = 1
        data = np.array([[age, sex, cp, trestbps, chol, thalach, oldpeak]])

        prediction = model.predict(data)

        if prediction[0] == 1:
            result = "🔴 High Risk of Heart Disease"
        else:
            result = "🟢 Low Risk of Heart Disease"

    return render(request,"home.html",{"result" : result})



