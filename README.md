# Pulse Prediction System using Machine Learning

## Overview

The Pulse Prediction System is a web-based application that analyzes cardiovascular indicators and predicts potential hypertension risk using a machine learning model.
The system calculates important clinical parameters such as Pulse Pressure (PP) and Mean Arterial Pressure (MAP), and then provides health insights and recommendations.

This project combines **Machine Learning**, **Flask**, and a **web interface** to simulate a basic clinical decision support tool.

---

## Features

* Predict hypertension risk using a trained Logistic Regression model
* Calculate Pulse Pressure (PP)
* Calculate Mean Arterial Pressure (MAP)
* Provide health alerts based on blood pressure ranges
* Generate recommendations based on the prediction results
* Interactive web interface

---

## Technologies Used

* Python
* Flask
* Scikit-learn
* NumPy
* HTML
* CSS
* Gunicorn (for deployment)

---

## Project Structure

project-folder
│
├── app.py
├── logreg_model.pkl
├── requirements.txt
│
├── templates
│   └── index.html
│
└── static
└── style.css

---

## Installation

1. Clone the repository

git clone https://github.com/your-username/pulse-prediction-system.git

2. Navigate to the project directory

cd pulse-prediction-system

3. Install dependencies

pip install -r requirements.txt

4. Run the application

python app.py

5. Open in browser

http://127.0.0.1:5000

---

## Deployment

This project can be deployed on cloud platforms such as Render using Gunicorn as the production server.

Start command for deployment:

gunicorn app:app

---

## Machine Learning Model

The system uses a **Logistic Regression model** trained on hypertension-related health data.
The model analyzes parameters such as systolic pressure, diastolic pressure, and heart rate to predict risk levels.

---

## Future Improvements

* Add more health indicators
* Improve model accuracy using larger datasets
* Add user authentication
* Store patient history in a database
* Visualize health metrics with charts

---

## Author

Veedushi Sahu

