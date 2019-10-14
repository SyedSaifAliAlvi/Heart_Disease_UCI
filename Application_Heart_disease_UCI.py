import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
warnings.filterwarnings("ignore", category=FutureWarning)
np.seterr(divide='ignore', invalid='ignore')
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
warnings.filterwarnings("ignore")
dt = pd.read_csv("heart.csv")
dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved','exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
dt['sex'][dt['sex'] == 0] = 'female'
dt['sex'][dt['sex'] == 1] = 'male'
#print(len(dt.columns))
dt['chest_pain_type'][dt['chest_pain_type'] == 0] = 'typical angina'
dt['chest_pain_type'][dt['chest_pain_type'] == 1] = 'atypical angina'
dt['chest_pain_type'][dt['chest_pain_type'] == 2] = 'non-anginal pain'
dt['chest_pain_type'][dt['chest_pain_type'] == 3] = 'asymptomatic'

dt['fasting_blood_sugar'][dt['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
dt['fasting_blood_sugar'][dt['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

dt['rest_ecg'][dt['rest_ecg'] == 0] = 'normal'
dt['rest_ecg'][dt['rest_ecg'] == 1] = 'ST-T wave abnormality'
dt['rest_ecg'][dt['rest_ecg'] == 2] = 'left ventricular hypertrophy'

dt['exercise_induced_angina'][dt['exercise_induced_angina'] == 0] = 'no'
dt['exercise_induced_angina'][dt['exercise_induced_angina'] == 1] = 'yes'

dt['st_slope'][dt['st_slope'] == 0] = 'upsloping'
dt['st_slope'][dt['st_slope'] == 1] = 'flat'
dt['st_slope'][dt['st_slope'] == 2] = 'downsloping'

dt['thalassemia'][dt['thalassemia'] == 1] = 'normal'
dt['thalassemia'][dt['thalassemia'] == 2] = 'fixed defect'
dt['thalassemia'][dt['thalassemia'] == 3] = 'reversable defect'

dt['sex'] = dt['sex'].astype('object')
dt['chest_pain_type'] = dt['chest_pain_type'].astype('object')
dt['fasting_blood_sugar'] = dt['fasting_blood_sugar'].astype('object')
dt['rest_ecg'] = dt['rest_ecg'].astype('object')
dt['exercise_induced_angina'] = dt['exercise_induced_angina'].astype('object')
dt['st_slope'] = dt['st_slope'].astype('object')
dt['thalassemia'] = dt['thalassemia'].astype('object')

dt = pd.get_dummies(dt,prefix=['st_slope'],columns=['st_slope'])
dt = pd.get_dummies(dt, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(dt.drop('target', 1), dt['target'], test_size = .271, random_state=37)
model = SVC(kernel='linear',gamma='scale',probability=True)
model.fit(X_train, y_train)

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

data = np.array([age,trestbps,chol,thalach,oldpeak,ca,st_slope_downsloping,st_slope_flat,st_slope_upsloping,sex,chest_pain_type_atypical_angina,chest_pain_type_non_anginal_pain,chest_pain_type_typical_angina,fbs,rest_ecg_left_ventricular_hypertrophy,rest_ecg_normal,exang,thalassemia_fixed_defect,thalassemia_normal,thalassemia_reversable_defect])
user_df = pd.DataFrame(data,['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved', 'st_depression', 'num_major_vessels', 'st_slope_downsloping', 'st_slope_flat', 'st_slope_upsloping', 'sex_male', 'chest_pain_type_atypical angina', 'chest_pain_type_non-anginal pain', 'chest_pain_type_typical angina', 'fasting_blood_sugar_lower than 120mg/ml', 'rest_ecg_left ventricular hypertrophy', 'rest_ecg_normal', 'exercise_induced_angina_yes', 'thalassemia_fixed defect', 'thalassemia_normal',
       'thalassemia_reversable defect']).transpose()

predict = model.predict(user_df)
res = predict[0]
if res== 0:
       print('Congratulations you DO NOT HAVE HEART DISEASE')
else:
       print('I am sorry to say but you have heart disease')