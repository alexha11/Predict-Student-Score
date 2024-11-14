import pandas as pd
import pickle

# Load model 
reg = pickle.load(open('model.pkl', 'rb'))

# Input data
hours_studied = input("Enter the number of hours studied: (0 -> 24) ")
previous_scores = input("Enter the previous scores: (0 -> 100) ")
extracurricular_activities = input("Enter the number of extracurricular activities: (0 for no and 1 for yes) ")
sleep_hours = input("Enter the number of sleep hours: (0 -> 24) ")
sample_question_papers_practiced = input("Enter the number of sample question papers practiced: ")

# create a new input data
new_data = {'Hours Studied': hours_studied, 'Previous Scores': previous_scores, 'Extracurricular Activities': extracurricular_activities ,'Sleep Hours': sleep_hours ,'Sample Question Papers Practiced': sample_question_papers_practiced}
new_df = pd.DataFrame([new_data])


# Predict data
performance = reg.predict(new_df)
print("Student performance ", performance)
