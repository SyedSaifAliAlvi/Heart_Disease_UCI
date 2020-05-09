import joblib
import numpy as np
import pandas as pd

model = joblib.load('weights_model.pkl')

print('Lets check your heart!!!!!')
#Application
print('Enter you age?')
age = input()
print('The persons resting blood pressure (mm Hg on admission to the hospital)')
trestbps = input()
print('The persons cholesterol measurement in mg/dl')
chol = input()
print('The persons maximum heart rate achieved')
thalach = input()
print('ST depression induced by exercise relative to rest  1- 4')
oldpeak = input()
print('The number of major vessels (0-3)')
ca = input()
print('the slope of the peak exercise ST segment (Value 0: upsloping, Value 1: flat, Value 2: downsloping)')
slope = input()
st_slope_downsloping = 0
st_slope_flat = 0
st_slope_upsloping = 0
if slope ==0:
    st_slope_downsloping=1
elif slope==1:
    st_slope_flat=1
elif slope==2:
    st_slope_upsloping=1
print('The persons sex (1 = male, 0 = female)')
sex = input()
print('The chest pain experienced (Value 0: typical angina, Value 1: atypical angina, Value 2: non-anginal pain, Value 3: asymptomatic)')
cp = input()
chest_pain_type_atypical_angina=0
chest_pain_type_non_anginal_pain=0
chest_pain_type_typical_angina=0
if cp ==1:
    chest_pain_type_atypical_angina = 1
elif cp==2:
    chest_pain_type_non_anginal_pain = 1
elif cp==3:
    chest_pain_type_typical_angina = 1
print('The persons fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)')
fbs = input()
print('Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes criteria)')
restecg = input()
rest_ecg_left_ventricular_hypertrophy=0
rest_ecg_normal=0
if restecg ==0:
    rest_ecg_normal=1
elif restecg==2:
    rest_ecg_left_ventricular_hypertrophy=1
print('Exercise induced angina (1 = yes; 0 = no)')
exang = input()
print('A blood disorder called thalassemia (0 = normal; 1 = fixed defect; 2 = reversable defect)')
thal = input()
thalassemia_fixed_defect = 0
thalassemia_normal=0
thalassemia_reversable_defect=0
if thal==0:
    thalassemia_normal=1
elif thal==1:
    thalassemia_normal=1
elif thal==2:
    thalassemia_reversable_defect=1

data = np.array([age,trestbps,chol,thalach,oldpeak,ca,st_slope_downsloping,
                 st_slope_flat,st_slope_upsloping,
                 sex,chest_pain_type_atypical_angina,
                 chest_pain_type_non_anginal_pain,
                 chest_pain_type_typical_angina,fbs,
                 rest_ecg_left_ventricular_hypertrophy,
                 rest_ecg_normal,exang,thalassemia_fixed_defect,
                 thalassemia_normal,thalassemia_reversable_defect])
user_df = pd.DataFrame(data,['age', 'resting_blood_pressure', 'cholesterol', 
                             'max_heart_rate_achieved', 'st_depression', 'num_major_vessels', 
                             'st_slope_downsloping', 'st_slope_flat',
                             'st_slope_upsloping', 'sex_male', 'chest_pain_type_atypical angina', 
                             'chest_pain_type_non-anginal pain', 
                             'chest_pain_type_typical angina', 
                             'fasting_blood_sugar_lower than 120mg/ml', 
                             'rest_ecg_left ventricular hypertrophy', 
                             'rest_ecg_normal', 'exercise_induced_angina_yes', 
                             'thalassemia_fixed defect', 
                             'thalassemia_normal',
       'thalassemia_reversable defect']).transpose()

predict = model.predict(user_df)
res = predict[0]
if res== 0:
       print('Congratulations you DO NOT HAVE HEART DISEASE')
else:
       print('I am sorry to say but you have heart disease')
