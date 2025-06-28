import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from tkinter import *
from PIL import Image, ImageTk

# --- Load and prepare data ---
heart_data = pd.read_csv('heart.csv')

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Apply SMOTE for class balancing
smote = SMOTE(random_state=2)
X_res, Y_res = smote.fit_resample(X, Y)

# Split and scale
X_train, X_test, Y_train, Y_test = train_test_split(X_res, Y_res, test_size=0.2, random_state=2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Train improved model ---
model = RandomForestClassifier(n_estimators=100, random_state=2)
model.fit(X_train, Y_train)

# --- Prediction Function ---
def predict_heart_disease():
    try:
        user_input = [
            float(entry_age.get()), float(entry_sex.get()), float(entry_cp.get()),
            float(entry_trestbps.get()), float(entry_chol.get()), float(entry_fbs.get()),
            float(entry_restecg.get()), float(entry_thalach.get()), float(entry_exang.get()),
            float(entry_oldpeak.get()), float(entry_slope.get()), float(entry_ca.get()), float(entry_thal.get())
        ]

        # Convert to DataFrame with correct column names
        input_df = pd.DataFrame([user_input], columns=X.columns)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)[0][1]

        # Display result in GUI
        if prediction[0] == 0:
            result_label.config(text=f"No Heart Disease\nConfidence: {prob:.2f}", fg="green")
        else:
            result_label.config(text=f"HAS Heart Disease\nConfidence: {prob:.2f}", fg="red")

        # Save to SQLite
        conn = sqlite3.connect('heart_predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                age REAL, sex REAL, cp REAL, trestbps REAL, chol REAL, fbs REAL,
                restecg REAL, thalach REAL, exang REAL, oldpeak REAL,
                slope REAL, ca REAL, thal REAL,
                prediction INTEGER, confidence REAL, timestamp TEXT
            )
        ''')
        cursor.execute('''
            INSERT INTO predictions (
                age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal,
                prediction, confidence, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (*user_input, int(prediction[0]), float(prob), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

    except Exception as e:
        result_label.config(text=f"Error: {str(e)}", fg="orange")

# --- GUI Setup ---
root = Tk()
root.title("Heart Disease Prediction System")
root.geometry("750x650")
root.resizable(False, False)

# Background Image
try:
    bg_img = Image.open("background.png")
    bg_img = bg_img.resize((750, 650))
    bg_photo = ImageTk.PhotoImage(bg_img)
    bg_label = Label(root, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
except Exception as e:
    print(f"Background image error: {e}")
    root.configure(bg="lightblue")

# Heart logo (optional)
try:
    heart_img = Image.open("heart.png")
    heart_img = heart_img.resize((80, 80))
    heart_photo = ImageTk.PhotoImage(heart_img)
    heart_label = Label(root, image=heart_photo, bg='lightblue')
    heart_label.place(x=335, y=10)
except:
    pass

# Glass Frame
glass_frame = Frame(root, bg="#ffffff", bd=0, highlightthickness=0)
glass_frame.place(x=50, y=110, width=650, height=480)
glass_frame.lift()

# Title
title_label = Label(root, text="Heart Disease Prediction", font=("Arial", 22, "bold"), bg="lightblue")
title_label.place(x=180, y=90)
title_label.lift()

# Form Fields
form_frame = Frame(glass_frame, bg="#ffffff")
form_frame.pack(pady=10)

labels = [
    "Age", "Sex (1=Male, 0=Female)", "Chest Pain Type (0-3)", "Resting Blood Pressure",
    "Cholesterol", "Fasting Blood Sugar (1/0)", "Rest ECG (0-2)",
    "Max Heart Rate", "Exercise Induced Angina (1/0)", "Oldpeak",
    "Slope (0-2)", "Number of Major Vessels (0-3)", "Thalassemia (3,6,7)"
]

entries = []
row = 0

for i, text in enumerate(labels):
    label = Label(form_frame, text=text, font=("Arial", 11), bg="#ffffff")
    entry = Entry(form_frame, width=20, bg="#f0f0f0", highlightbackground="lightblue", highlightthickness=1)

    label.grid(row=row, column=(i % 2) * 2, padx=10, pady=5, sticky=W)
    entry.grid(row=row, column=(i % 2) * 2 + 1, padx=10, pady=5)

    entries.append(entry)
    if i % 2 == 1:
        row += 1

(entry_age, entry_sex, entry_cp, entry_trestbps, entry_chol,
 entry_fbs, entry_restecg, entry_thalach, entry_exang,
 entry_oldpeak, entry_slope, entry_ca, entry_thal) = entries

# Predict Button
predict_button = Button(glass_frame, text="Predict Heart Disease", command=predict_heart_disease,
                        font=("Arial", 14), bg="darkblue", fg="white", width=25)
predict_button.pack(pady=20)

# Result Label
result_label = Label(glass_frame, text="", font=("Arial", 16, "bold"), bg="#ffffff")
result_label.pack(pady=10)

# Start App
root.mainloop()
