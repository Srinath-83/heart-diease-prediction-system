# 🫀 Heart Disease Prediction System

A Python-based heart disease prediction system using machine learning (Logistic Regression), with a graphical interface built in Tkinter. User inputs are logged into a local SQLite database for future reference.

## Features

- *Machine Learning Model**: Trained using the UCI Heart Disease dataset with Logistic Regression.
- *Tkinter GUI**: Simple form for users to input medical data and get predictions.
- *Real-time Predictions**: Tells whether the person is likely to have heart disease or not.
- *SQLite Logging**: Every prediction is stored locally with all inputs and timestamp.
- *UI Enhancement Support**: Background image and heart logo can be added optionally.



## 🛠️ Tech Stack

| Component        | Tool                  |
|------------------|------------------------|
| Language         | Python                 |
| ML Library       | Scikit-learn           |
| Data Handling    | Pandas, NumPy          |
| GUI              | Tkinter                |
| Database         | SQLite3                |
| Image Handling   | PIL (Pillow)           |



## Files Included

- `heart_disease.py` - Main script
- `heart.csv` - Dataset used (ensure it's present in the same directory)




## How to Run

1. **Install requirements (if needed):**

   ```bash
   pip install pandas numpy scikit-learn pillow
