from flask import Flask, request, render_template, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from flask_migrate import Migrate
from sklearn.datasets import make_regression 
import pickle
import math

import csv
import warnings
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pymysql
from sqlalchemy.orm import sessionmaker
import logging  # Ensure this line is included at the top

# ... your other imports

# Configure logging
logging.basicConfig(level=logging.INFO)

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Flask app
app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:@localhost/crime_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initialize Bcrypt for password hashing
bcrypt = Bcrypt(app)

# Initialize Flask-Migrate for database migrations
migrate = Migrate(app, db)

# Create a session factory within an application context
def create_session():
    global session
    Session = sessionmaker(bind=db.engine)
    session = Session()
X_train, y_train = make_regression(n_samples=100, n_features=4, random_state=42)
# Load your model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
with open('Model/model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Define your routes
@app.route('/req.html')
def req():
    return render_template('req.html')

# Define other routes and models here

# Ensure the session is created before handling requests
with app.app_context():
    create_session()

@app.route('/submit_request', methods=['POST'])
def submit_request():
    crime_data = request.form['crime_data']
    conn = pymysql.connect(host='localhost', user='your_username', password='your_password', database='crime_db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO prediction_requests (data, submission_date) VALUES (%s, NOW())', (crime_data,))
    conn.commit()
    conn.close()
    return redirect(url_for('view_results'))

@app.route('/view_results')
def view_results():
    conn = pymysql.connect(host='localhost', user='your_username', password='your_password', database='crime_db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM prediction_requests')
    prediction_requests = cursor.fetchall()
    conn.close()
    return render_template('view_results.html', prediction_requests=prediction_requests)

@app.route('/prediction_result/<int:request_id>')
def prediction_result(request_id):
    conn = pymysql.connect(host='localhost', user='your_username', password='your_password', database='crime_db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM prediction_requests WHERE id = %s', (request_id,))
    request_data = cursor.fetchone()
    conn.close()
    prediction_result = f"Prediction result for request {request_id}: [Dummy Result]"
    return render_template('req.html', prediction_result=prediction_result)







class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)

    def __init__(self, name, email, message):
        self.name = name
        self.email = email
        self.message = message

@app.route('/contact', methods=['POST'])
def contact():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']

    new_contact = Contact(name=name, email=email, message=message)
    db.session.add(new_contact)
    db.session.commit()

    return redirect(url_for('home'))


# Upload page (CSV upload for chart display)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Upload page (CSV upload for chart display)
@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'dataFile' not in request.files:
        return redirect(url_for('updata'))  # Redirect to the updata page
    
    file = request.files['dataFile']
    if file.filename == '':
        return redirect(url_for('updata'))  # Redirect to the updata page if no file is selected
    
    if file and file.filename.endswith('.csv'):
        file.save(os.path.join(UPLOAD_FOLDER, 'crime_data.csv'))
        return redirect(url_for('chart_page'))  # Redirect to the chart page after a successful upload
    
    return redirect(url_for('updata'))  # Redirect to the updata page in case of an error

# API route to provide data to the chart (for chart.html)
@app.route('/api/crime-rate-data')
def crime_rate_data():
    data = {'cities': [], 'crimeRates': {}}

    with open(os.path.join(UPLOAD_FOLDER, 'crime_data.csv'), 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            city = row[2]  # City column
            if city in data['crimeRates']:
                data['crimeRates'][city] += 1
            else:
                data['crimeRates'][city] = 1

    # Separate the city names and counts into two lists
    data['cities'] = list(data['crimeRates'].keys())
    data['crimeRates'] = list(data['crimeRates'].values())

    return jsonify(data)


# Route to render the chart page (chart.html)
@app.route('/chart.html')
def chart_page():
    return render_template("chart.html")  # Serve the chart.html template

# Route to render the updata page (updata.html)
@app.route('/updata.html')
def updata():
    return render_template("updata.html") 





@app.route('/')
def home():
    return render_template("home.html")
    
users = []

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size (16 MB)

# Ensure the uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/data_management.html')
def data_management():
    return render_template('data_management.html')

@app.route('/upload_data', methods=['POST'])
def upload_data():
    if 'data_file' not in request.files:
        return 'No file part in the request', 400
    file = request.files['data_file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('preview_data', filename=file.filename))
        except Exception as e:
            return f'An error occurred: {e}', 500
    return 'An unexpected error occurred', 500

@app.route('/preview/<filename>')
def preview_data(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        data = pd.read_excel(filepath)  # Read the Excel file
        data_html = data.to_html()  # Convert the data to HTML
        return render_template('preview_data.html', table=data_html)
    except Exception as e:
        return f'An error occurred: {e}', 500

@app.route('/management.html')
def management():
    return render_template("management.html", users=users)

@app.route('/add_user', methods=['POST'])
def add_user():
    username = request.form['username']
    email = request.form['email']
    # Check if user already exists
    if any(user['username'] == username for user in users):
        return redirect(url_for('management', message='User already exists'))
    users.append({'username': username, 'email': email})
    return redirect(url_for('management'))

@app.route('/edit_user/<username>', methods=['POST'])
def edit_user(username):
    new_username = request.form['username']
    new_email = request.form['email']
    for user in users:
        if user['username'] == username:
            user['username'] = new_username
            user['email'] = new_email
            break
    return redirect(url_for('management'))

@app.route('/delete_user/<username>', methods=['POST'])
def delete_user(username):
    global users
    users = [user for user in users if user['username'] != username]
    return redirect(url_for('management'))


@app.route('/index.html')
def index():
    return render_template("index.html")



class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    mobile = db.Column(db.String(15), nullable=False)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

@app.route('/a_signup.html', methods=['GET', 'POST'])
def a_signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        mobile = request.form['mobile']
        password = request.form['password']
        confirm_password = request.form['confirm-password']
        
        if password != confirm_password:
            return "Passwords do not match", 400
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_admin = Admin(name=name, email=email, mobile=mobile, password=hashed_password)
        
        db.session.add(new_admin)
        db.session.commit()
        
        return redirect(url_for('a_login'))
    
    return render_template('a_signup.html')

@app.route('/a_login.html', methods=['GET', 'POST'])
def a_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        admin = Admin.query.filter_by(email=email).first()
        if admin and check_password_hash(admin.password, password):
            return redirect(url_for('dashboard'))  # Redirect to dashboard or another page
        else:
            return "Invalid email or password", 401
    
    return render_template('a_login.html')

class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    mobile = db.Column(db.String(15), nullable=False)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

@app.route('/u_signup.html', methods=['GET', 'POST'])
def u_signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        mobile = request.form['mobile']
        password = request.form['password']
        confirm_password = request.form['confirm-password']
        
        if password != confirm_password:
            return "Passwords do not match", 400
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = Users(name=name, email=email, mobile=mobile, password=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for('u_login'))
    
    return render_template('u_signup.html')

@app.route('/u_login.html', methods=['GET', 'POST'])
def u_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            return redirect(url_for('dashboard'))  # Redirect to dashboard or another page
        else:
            return "Invalid email or password", 401
    
    return render_template('u_login.html')


@app.route('/dashboard.html')
def dashboard():
    return render_template("dashboard.html")

@app.route('/u_dashboard.html')
def u_dashboard():
    return render_template("u_dashboard.html")    

app.secret_key = 'your_secret_key'  # Required for session management and flash messages

# Sample user data


class Profile(db.Model):
    __tablename__ = 'profiles'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    phone = db.Column(db.String(20))
    profile_pic = db.Column(db.String(255))

@app.route('/u/profile/<int:profile_id>', methods=['GET', 'POST'])
def u_profile(profile_id):
    # Query the profile based on the profile_id
    profile = db.session.get(Profile, profile_id)

    # If the profile does not exist, create a new one
    if profile is None:
        if request.method == 'POST':
            name = request.form.get('name')
            email = request.form.get('email')
            phone = request.form.get('phone')
            profile_pic = request.files.get('profile_pic')

            profile_pic_path = 'uploads/default.png'  # Default profile picture
            if profile_pic:
                filename = secure_filename(profile_pic.filename)
                profile_pic_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                profile_pic.save(profile_pic_path)

            # Create a new profile with the provided information
            profile = Profile(
                id=profile_id,
                name=name,
                email=email,
                phone=phone,
                profile_pic=profile_pic_path
            )
            db.session.add(profile)
            db.session.commit()
            return redirect(url_for('u_profile', profile_id=profile.id))
        return render_template('u_profile.html', profile=None)

    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        profile_pic = request.files.get('profile_pic')

        if profile_pic:
            filename = secure_filename(profile_pic.filename)
            profile_pic_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            profile_pic.save(profile_pic_path)
            profile.profile_pic = profile_pic_path

        profile.name = name
        profile.email = email
        profile.phone = phone

        db.session.commit()
        return redirect(url_for('u_profile', profile_id=profile.id))

    return render_template('u_profile.html', profile=profile)

@app.route('/a/profile')
def a_profile():
    profiles = Profile.query.all()
    return render_template('a_profile.html', profiles=profiles)

@app.route('/a/u/<int:profile_id>/edit', methods=['GET', 'POST'])
def edit_profile(profile_id):
    profile = db.session.get(Profile, profile_id)
    
    if profile is None:
        return "Profile not found", 404

    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        profile_pic = request.files.get('profile_pic')

        profile_pic_path = 'uploads/default.png'
        if profile_pic:
            filename = secure_filename(profile_pic.filename)
            profile_pic_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            profile_pic.save(profile_pic_path)

        profile.name = name
        profile.email = email
        profile.phone = phone
        profile.profile_pic = profile_pic_path

        db.session.commit()
        return redirect(url_for('a_profile'))

    return render_template('a_edit_profile.html', profile=profile)

@app.route('/a/u/<int:profile_id>/delete')
def delete_profile(profile_id):
    profile = db.session.get(Profile, profile_id)
    if profile:
        db.session.delete(profile)
        db.session.commit()
    return redirect(url_for('a_profile'))

class ClientFeedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    feedback_text = db.Column(db.Text, nullable=False)
    date = db.Column(db.DateTime, default=db.func.current_timestamp())

@app.route('/client_feedback.html', methods=['GET', 'POST'])
def client_feedback():
    if request.method == 'POST':
        feedback_text = request.form['feedback_text']
        if feedback_text:
            new_feedback = ClientFeedback(feedback_text=feedback_text)
            db.session.add(new_feedback)
            db.session.commit()
        return redirect(url_for('client_feedback'))
    
    return render_template("client_feedback.html")

@app.route('/display_feedback.html')
def display_feedback():
    feedback_list = ClientFeedback.query.all()
    return render_template("display_feedback.html", feedback_list=feedback_list)

prediction_requests = [
    {"id": 1, "data": "Data for request 1", "date": "2024-09-16"},
    {"id": 2, "data": "Data for request 2", "date": "2024-09-17"}
]

results = {
    1: "Prediction result for request 1.",
    2: "Prediction result for request 2."
}


@app.route('/submit_prediction_request', methods=['POST'])
def submit_prediction_request():
    crime_data = request.form['crime_data']
    
    if crime_data:
        # Simulate saving the request and processing the prediction
        # For demonstration purposes, add a new dummy request and result
        new_id = len(prediction_requests) + 1
        prediction_requests.append({
            "id": new_id,
            "data": crime_data,
            "date": "2024-09-18"  # Dummy date
        })
        results[new_id] = "This is a dummy result based on the provided data."
        return redirect(url_for('req'))
    else:
        return redirect(url_for('req'))



@app.route('/view_prediction_result/<int:request_id>')
def view_prediction_result(request_id):
    # Fetch the result for a specific prediction request
    request_data = next((item['data'] for item in prediction_requests if item['id'] == request_id), "Data not found")
    result = results.get(request_id, "Result not available")
    return render_template('view_prediction_result.html', request_data=request_data, result=result)





@app.route('/api/crime-rate-data', methods=['GET'])
def get_crime_rate_data():
    # Example static data; replace with dynamic data from your model/database as needed
    data = {
        'cities': ['Ahmedabad', 'Bengaluru', 'Chennai', 'Delhi', 'Mumbai'],
        'crimeRates': [120, 180, 90, 210, 160]  # Example static data for demonstration
    }
    return jsonify(data)

@app.route('/chart.html')
def chart():
    return render_template('chart.html')

@app.route('/predict', methods=['POST'])
def predict_result():
    city_names = {
        '0': 'Ahmedabad', '1': 'Bengaluru', '2': 'Chennai', '3': 'Coimbatore',
        '4': 'Delhi', '5': 'Ghaziabad', '6': 'Hyderabad', '7': 'Indore',
        '8': 'Jaipur', '9': 'Kanpur', '10': 'Kochi', '11': 'Kolkata',
        '12': 'Kozhikode', '13': 'Lucknow', '14': 'Mumbai', '15': 'Nagpur',
        '16': 'Patna', '17': 'Pune', '18': 'Surat'
    }

    crimes_names = {
        '0': 'Crime Committed by Juveniles', '1': 'Crime against SC', '2': 'Crime against ST',
        '3': 'Crime against Senior Citizen', '4': 'Crime against children', '5': 'Crime against women',
        '6': 'Cyber Crimes', '7': 'Economic Offences', '8': 'Kidnapping', '9': 'Murder'
    }

    population = {
        '0': 63.50, '1': 85.00, '2': 87.00, '3': 21.50, '4': 163.10, '5': 23.60, '6': 77.50,
        '7': 21.70, '8': 30.70, '9': 29.20, '10': 21.20, '11': 141.10, '12': 20.30,
        '13': 29.00, '14': 184.10, '15': 25.00, '16': 20.50, '17': 50.50, '18': 45.80
    }

    city_code = request.form["city"]
    crime_code = request.form['crime']
    year = request.form['year']
    pop = population[city_code]

    # Adjust population based on the year
    year_diff = int(year) - 2011
    pop = pop + 0.01 * year_diff * pop

    crime_rate = model.predict([[year, city_code, pop, crime_code]])[0]

    city_name = city_names[city_code]
    crime_type = crimes_names[crime_code]

    if crime_rate <= 1:
        crime_status = "Very Low Crime Area"
    elif crime_rate <= 5:
        crime_status = "Low Crime Area"
    elif crime_rate <= 15:
        crime_status = "High Crime Area"
    else:
        crime_status = "Very High Crime Area"

    cases = math.ceil(crime_rate * pop)

    return render_template('result.html', city_name=city_name, crime_type=crime_type, year=year,
                           crime_status=crime_status, crime_rate=crime_rate, cases=cases, population=pop)

@app.route('/api/analysis')
def get_analysis():
    # Load and process data
    data = pd.read_csv('data/crime_data.csv')
    processed_data = preprocess_data(data)
    _, correlation_matrix, trend = analyze_data(processed_data)

    # Convert results to JSON
    correlation_json = correlation_matrix.to_dict()
    trend_json = trend.to_dict()

    return jsonify({
        'correlation_matrix': correlation_json,
        'trend': trend_json
    })

@app.route('/analysis')
def analysis_page():
    return render_template('analysis.html')

if __name__ == '__main__':
    app.run(debug=False)
