from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib
import json
from datetime import datetime, timedelta
import os

# Function to generate a random secret key
def generate_secret_key():
    return os.urandom(24).hex()

# Set up the Flask application
app = Flask(__name__)
app.secret_key = generate_secret_key()  # Generate and set a secret key

# Load the trained models and scaler
model_class = joblib.load('predictive_model_class.pkl')
model_reg = joblib.load('predictive_model_reg.pkl')
scaler = joblib.load('scaler.pkl')

# User data file
USER_DATA_FILE = 'users.json'
PASSBOOK_DATA_FILE = 'passbook.json'

def load_users():
    try:
        with open(USER_DATA_FILE, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_users(users):
    with open(USER_DATA_FILE, 'w') as file:
        json.dump(users, file)

def load_passbook():
    try:
        with open(PASSBOOK_DATA_FILE, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_passbook(passbook_entries):
    with open(PASSBOOK_DATA_FILE, 'w') as file:
        json.dump(passbook_entries, file)

@app.route('/')
def index():
    machine_no = session.get('machine_no')
    return render_template('index.html', machine_no=machine_no)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        machine_no = request.form['machine_no']
        password = request.form['password']
        
        users = load_users()
        if machine_no in users and users[machine_no] == password:
            session['machine_no'] = machine_no
            return redirect(url_for('passbook'))
        else:
            return 'Invalid machine number or password', 401

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        machine_no = request.form['machine_no']
        password = request.form['password']
        
        users = load_users()
        if machine_no not in users:
            users[machine_no] = password
            save_users(users)
            return redirect(url_for('login'))
        else:
            return 'Machine number already exists', 400

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('machine_no', None)
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'machine_no' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        input_data = request.form.to_dict()
        input_df = pd.DataFrame([input_data])
        
        # Convert input data to float
        input_df = input_df.astype(float)
        
        # Preprocess input data using the saved scaler
        input_processed = scaler.transform(input_df)
        
        # Predict maintenance needs
        prediction_class = model_class.predict(input_processed)
        
        # Predict expected days to maintenance
        prediction_reg = model_reg.predict(input_processed)
        
        # Generate a user-friendly message and expected days
        if prediction_class[0] == 1:
            message = "Yes, the machine needs preventive maintenance."
        else:
            message = "No need for preventive maintenance now."
        
        expected_days = int(prediction_reg[0])
        operating_hours = int(input_data.get('operating_hours', 0))
        save_to_passbook(message, expected_days, operating_hours)

        return redirect(url_for('result', prediction=message, expected_days=expected_days))

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    expected_days = request.args.get('expected_days')
    return render_template('result.html', prediction=prediction, expected_days=expected_days)

@app.route('/passbook')
def passbook():
    if 'machine_no' not in session:
        return redirect(url_for('login'))

    passbook_entries = load_passbook().get(session['machine_no'], [])
    return render_template('passbook.html', events=passbook_entries, machine_no=session['machine_no'])

@app.route('/schedule')
def schedule():
    if 'machine_no' not in session:
        return redirect(url_for('login'))

    passbook_entries = load_passbook().get(session['machine_no'], [])
    future_dates = []
    
    for entry in passbook_entries:
        maintenance_date = datetime.strptime(entry['date'], "%Y-%m-%d") + timedelta(days=int(entry['days_to_maintenance']))
        future_dates.append(maintenance_date.strftime("%Y-%m-%d"))
    
    return render_template('schedule.html', future_dates=future_dates[:20])

def save_to_passbook(description, days_to_maintenance, operating_hours):
    date = datetime.now().strftime("%Y-%m-%d")
    entry = {
        "date": date,
        "machine_no": session['machine_no'],
        "description": description,
        "operating_hours": operating_hours,
        "days_to_maintenance": days_to_maintenance
    }

    passbook_entries = load_passbook()
    if session['machine_no'] not in passbook_entries:
        passbook_entries[session['machine_no']] = []

    passbook_entries[session['machine_no']].append(entry)
    save_passbook(passbook_entries)

if __name__ == '__main__':
    app.run(debug=True)
