# 🎓 Student Marks Predictor

A lightweight ML web app that predicts a student's exam score using five key factors.  
Built with **Flask** + **scikit-learn** (Linear Regression) and a clean HTML/CSS/JS front-end.

---

## Features

| Feature | Details |
|---|---|
| ML Model | Linear Regression (scikit-learn) |
| Accuracy | R² ≈ 0.92, MAE ≈ 3.4 points |
| Backend | Flask (Python) |
| Frontend | Plain HTML + CSS + vanilla JS |
| Inputs | Study hours, Attendance, Previous score, Sleep hours, Assignments done |
| Output | Predicted score (0–100) + Grade (A+ to F) |

---

## Project Structure

```
student-marks-prediction/
├── app.py                   # Flask web server
├── generate_and_train.py    # Data generation + model training
├── requirements.txt
├── model/
│   ├── student_data.csv     # Generated dataset (1000 rows)
│   ├── marks_model.pkl      # Trained Linear Regression model
│   └── scaler.pkl           # StandardScaler
└── templates/
    └── index.html           # Web UI
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (run once)
```bash
python generate_and_train.py
```
Expected output:
```
Dataset saved  -> model/student_data.csv  (1000 rows)
Model trained  ->  MAE: 3.36  |  R2: 0.9204
Saved          -> model/marks_model.pkl  &  model/scaler.pkl
```

### 3. Start the web app
```bash
python app.py
```
Open your browser at **http://127.0.0.1:5000**

---

## API

**POST** `/predict`

```json
{
  "study_hours": 8,
  "attendance": 90,
  "prev_score": 75,
  "sleep_hours": 7,
  "assignments_done": 85
}
```

Response:
```json
{
  "predicted_marks": 98.1,
  "grade": "A+",
  "color": "#16a34a"
}
```

---

## Grade Scale

| Grade | Score Range |
|---|---|
| A+ | 90 – 100 |
| A  | 80 – 89  |
| B  | 70 – 79  |
| C  | 60 – 69  |
| D  | 50 – 59  |
| F  | 0  – 49  |
