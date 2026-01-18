import secrets
import warnings
from collections import defaultdict
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, redirect, url_for, session, jsonify, make_response, flash
import firebase_admin
from firebase_admin import credentials, db, auth, storage
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
import requests
import io
import csv
import os
from dotenv import load_dotenv
from collections import Counter

#f
# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

# Firebase configuration from environment variables
cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
database_url = os.getenv("FIREBASE_DATABASE_URL")
API_KEY = os.getenv("FIREBASE_API_KEY")

# Initialize Firebase
try:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': database_url
    })
    ref = db.reference('/')
except Exception as e:
    logger.error(f"Failed to initialize Firebase: {str(e)}")
    raise


@app.route('/')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    user_data = ref.child(f"users/{session['user']}").get() or {}
    return render_template('index.html', user_data=user_data)


@app.route('/login', methods=['GET', 'POST'])
def login():
    user_data = {}
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }

            response = requests.post(url, json=payload)
            data = response.json()
            logger.debug(f"Login response: {data}")

            if 'idToken' in data:
                uid = data['localId']
                session['user'] = uid
                session['email'] = email

                user_ref = ref.child(f"users/{uid}")
                user_data = user_ref.get() or {}

                session['role'] = user_data.get('role', 'user')
                session.modified = True

                if not user_data.get('email'):
                    user_ref.update({'email': email})

                if user_data.get('display_name'):
                    session['display_name'] = user_data['display_name']

                return redirect(url_for('dashboard'))
            else:
                error = data.get('error', {}).get('message', 'Unknown error')
                return render_template('login.html', error=f"Login failed: {error}")

        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return render_template('login.html', error="Unexpected error during login")

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    user_data = {}

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }

            response = requests.post(url, json=payload)
            data = response.json()
            logger.debug(f"Signup response: {data}")

            if 'idToken' in data:
                uid = data['localId']
                session['user'] = uid
                session['email'] = email

                user_ref = ref.child(f"users/{uid}")
                user_data = user_ref.get() or {}
                if not user_data.get('email'):
                    user_ref.update({'email': email})

                return redirect(url_for('dashboard'))
            else:
                error = data.get('error', {}).get('message', 'Unknown error')
                return render_template('signup.html', error=f"Signup failed: {error}")

        except Exception as e:
            logger.error(f"Signup error: {str(e)}")
            return render_template('signup.html', error="Unexpected error during signup")

    return render_template('signup.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model_columns.json', 'r') as f:
            model_columns = json.load(f)

        product_name = request.form.get('product_name', 'Unnamed Product')

        data = {
            'Product_ID': request.form['product_id'],
            'Category': request.form['category'],
            'Price': float(request.form['price']),
            'Units_Sold': float(request.form['units_sold']),
            'Promotion': 1 if request.form['promotion'] == 'Yes' else 0,
            'Holiday_Flag': 1 if request.form['holiday_flag'] == 'Yes' else 0,
            'Competitor_Price': float(request.form['competitor_price']),
            'Special_Events': request.form['special_events'],
            'Day_of_Week': request.form['day_of_week'],
            'Month': int(request.form['month'])
        }

        if data['Price'] < 0 or data['Competitor_Price'] < 0 or data['Units_Sold'] < 0:
            forecasts = ref.child(f"forecasts/{session['user']}").get() or {}
            return render_template('index.html',
                                   prediction_text="Error: Negative values not allowed.",
                                   forecasts=forecasts)

        if data['Month'] not in range(1, 13):
            forecasts = ref.child(f"forecasts/{session['user']}").get() or {}
            return render_template('index.html',
                                   prediction_text="Error: Month must be 1-12.",
                                   forecasts=forecasts)

        df = pd.DataFrame([data])
        df = pd.get_dummies(df)
        df = df.reindex(columns=model_columns, fill_value=0)

        prediction = np.expm1(model.predict(df))[0]

        uid = session['user']
        forecast_data = {
            "predictedRevenue": prediction,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "Category": data['Category'],
            "ProductName": product_name,
            "Price": data['Price'],
            "Units_Sold": data['Units_Sold'],
            "Promotion": data['Promotion'],
            "Holiday_Flag": data['Holiday_Flag']
        }

        ref.child(f"forecasts/{uid}/{data['Product_ID']}").set(forecast_data)

        dates = [
            datetime.now().strftime('%Y-%m-%d'),
            (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        ]

        predictions = [prediction, prediction * 1.05]

        prediction_data = {
            'prediction': prediction,
            'dates': dates,
            'predictions': predictions,
            'product_name': product_name
        }

        session['prediction_data'] = prediction_data

        return redirect(url_for('dashboard'))

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        forecasts = ref.child(f"forecasts/{session['user']}").get() or {}
        return render_template('index.html',
                               prediction_text=f"Error: {str(e)}",
                               forecasts=forecasts)


@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    uid = session['user']
    forecasts = ref.child(f"forecasts/{uid}").get() or {}
    user_data = ref.child(f"users/{uid}").get() or {}

    prediction_data = session.get('prediction_data')

    if not prediction_data and forecasts:
        latest_product = max(forecasts.keys())
        latest_prediction = float(forecasts[latest_product]['predictedRevenue'])

        dates = [
            forecasts[latest_product]['date'],
            (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        ]

        predictions = [latest_prediction, latest_prediction * 1.05]

        prediction_data = {
            'prediction': latest_prediction,
            'dates': dates,
            'predictions': predictions
        }

    total_predictions = len(forecasts)
    total_revenue = sum(float(f.get('predictedRevenue', 0)) for f in forecasts.values())
    highest_predicted = max((float(f.get('predictedRevenue', 0)) for f in forecasts.values()), default=0)
    avg_predicted_revenue = total_revenue / total_predictions if total_predictions > 0 else 0

    category_counts = {}
    for f in forecasts.values():
        category = f.get('Category', 'Unknown')
        revenue = float(f.get('predictedRevenue', 0))
        category_counts[category] = category_counts.get(category, 0) + revenue

    category_labels = list(category_counts.keys())
    category_values = list(category_counts.values())

    order_channel_dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(2)]

    order_channel_data = {}
    for f in forecasts.values():
        category = f.get('Category', 'Unknown')
        revenue = float(f.get('predictedRevenue', 0))
        if category in order_channel_data:
            order_channel_data[category] = [
                order_channel_data[category][0] + revenue * 0.5,
                order_channel_data[category][1] + revenue * 0.5
            ]
        else:
            order_channel_data[category] = [revenue * 0.5, revenue * 0.5]

    order_channel_others = order_channel_data.get('Home', [0, 0])
    order_channel_mobile = order_channel_data.get('Electronics', [0, 0])
    order_channel_website = order_channel_data.get('Clothing', [0, 0])

    if prediction_data:
        upper_bounds = [p * 1.1 for p in prediction_data['predictions']]
        lower_bounds = [p * 0.9 for p in prediction_data['predictions']]
    else:
        upper_bounds = [0, 0]
        lower_bounds = [0, 0]

    message = request.args.get('message')

    return render_template(
        'dashboard.html',
        prediction_data=prediction_data,
        total_predictions=total_predictions,
        total_revenue=total_revenue,
        highest_predicted=highest_predicted,
        avg_predicted_revenue=avg_predicted_revenue,
        category_labels=category_labels,
        category_values=category_values,
        order_channel_dates=order_channel_dates,
        order_channel_others=order_channel_others,
        order_channel_mobile=order_channel_mobile,
        order_channel_website=order_channel_website,
        upper_bounds=upper_bounds,
        lower_bounds=lower_bounds,
        forecasts=forecasts,
        message=message,
        user_data=user_data
    )


@app.before_request
def check_session():
    if request.endpoint in ['delete_prediction'] and 'user' not in session:
        return jsonify({'error': 'Session expired'}), 401


@app.route('/delete_prediction/<product_id>', methods=['DELETE', 'POST'])
def delete_prediction(product_id):
    return _delete_prediction(product_id)


def _delete_prediction(product_id):
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        prediction_ref = ref.child(f"forecasts/{session['user']}/{product_id}")
        if not prediction_ref.get():
            return jsonify({'error': 'Prediction not found'}), 404

        prediction_ref.delete()
        return jsonify({'success': True}), 200

    except Exception as e:
        logger.error(f"Delete failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/export_dashboard', methods=['POST'])
def export_dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    uid = session['user']
    forecasts = ref.child(f"forecasts/{uid}").get() or {}

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Product ID', 'Product Name', 'Predicted Revenue', 'Date', 'Category'])

    for pid, data in forecasts.items():
        writer.writerow([
            pid,
            data.get('ProductName', 'Unnamed Product'),
            data.get('predictedRevenue', 0),
            data.get('date', 'N/A'),
            data.get('Category', 'N/A')
        ])

    output.seek(0)
    return make_response(output.getvalue(), 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': 'attachment; filename=dashboard_report.csv'
    })


@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))


@app.route('/export_report', methods=['POST'])
def export_report():
    if 'user' not in session:
        return redirect(url_for('login'))

    forecasts = ref.child(f"forecasts/{session['user']}").get() or {}

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Product ID', 'Product Name', 'Predicted Revenue', 'Date', 'Category'])

    for pid, data in forecasts.items():
        writer.writerow([
            pid,
            data.get('ProductName', 'Unnamed Product'),
            data.get('predictedRevenue', 0),
            data.get('date', ''),
            data.get('Category', 'Unknown')
        ])

    output.seek(0)
    return make_response(output.getvalue(), 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': 'attachment; filename=report.csv'
    })


@app.route('/export_csv')
def export_csv():
    if 'user' not in session:
        return redirect(url_for('login'))

    uid = session['user']
    forecasts = ref.child(f"forecasts/{uid}").get() or {}

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Product ID', 'Product Name', 'Predicted Revenue', 'Date', 'Category'])

    for product_id, data in forecasts.items():
        writer.writerow([
            product_id,
            data.get('ProductName', 'Unnamed Product'),
            data.get('predictedRevenue', 0),
            data.get('date', ''),
            data.get('Category', 'Unknown')
        ])

    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=forecasts.csv'
    response.headers['Content-type'] = 'text/csv'
    return response


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'user' not in session:
        return redirect(url_for('login'))

    uid = session['user']
    user_data = ref.child(f"users/{uid}").get() or {}
    error = None

    if request.method == 'POST':
        updates = {}
        current_data = ref.child(f"users/{uid}").get() or {}
        current_email = current_data.get('email', session.get('email'))

        if 'update_profile' in request.form:
            display_name = request.form.get('display_name')
            company_name = request.form.get('company_name')

            if display_name:
                updates['display_name'] = display_name
            if company_name:
                updates['company_name'] = company_name

        if 'notifications' in request.form:
            notifications = {
                "email": 'email' in request.form,
                "sms": 'sms' in request.form
            }
            updates["notifications"] = notifications

        if 'update_email' in request.form:
            new_email = request.form.get('new_email')
            if new_email and current_email != new_email:
                try:
                    auth.update_user(uid, email=new_email)
                    updates["email"] = new_email
                    session['email'] = new_email
                except Exception as e:
                    error = f"Error updating email: {str(e)}"

        if 'update_password' in request.form:
            new_password = request.form.get('new_password')
            if new_password:
                try:
                    auth.update_user(uid, password=new_password)
                except Exception as e:
                    error = f"Error updating password: {str(e)}"

        if 'clear_forecasts' in request.form:
            try:
                ref.child(f"forecasts/{uid}").delete()
            except Exception as e:
                error = f"Error clearing forecasts: {str(e)}"

        if 'export_data' in request.form:
            try:
                user_data_export = {
                    'user_info': current_data,
                    'forecasts': ref.child(f"forecasts/{uid}").get() or {}
                }
                output = io.StringIO()
                json.dump(user_data_export, output, indent=2)
                output.seek(0)
                return make_response(output.getvalue(), 200, {
                    'Content-Type': 'application/json',
                    'Content-Disposition': 'attachment; filename=user_data.json'
                })
            except Exception as e:
                error = f"Error exporting data: {str(e)}"

        if updates:
            try:
                ref.child(f"users/{uid}").update(updates)
            except Exception as e:
                error = f"Error saving settings: {str(e)}"

        if error:
            return render_template('settings.html', user_data=user_data, error=error)

        return redirect(url_for('settings', message="Settings updated successfully"))

    return render_template('settings.html', user_data=user_data, error=error)


@app.route('/generate_api_key', methods=['POST'])
def generate_api_key():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    uid = session['user']
    try:
        api_key = secrets.token_urlsafe(32)
        ref.child(f"users/{uid}/api_key").set(api_key)
        return jsonify({'api_key': api_key})
    except Exception as e:
        return jsonify({'error': 'Failed to generate API key'}), 500


@app.route('/regenerate_api_key', methods=['POST'])
def regenerate_api_key():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    uid = session['user']
    try:
        api_key = secrets.token_urlsafe(32)
        ref.child(f"users/{uid}/api_key").set(api_key)
        return jsonify({'api_key': api_key})
    except Exception as e:
        return jsonify({'error': 'Failed to regenerate API key'}), 500


@app.route('/admin')
def admin():
    if 'user' not in session:
        return redirect(url_for('login'))

    uid = session['user']
    user_data = ref.child(f"users/{uid}").get() or {}

    session['role'] = user_data.get('role', 'user')

    if session['role'] != 'admin':
        return redirect(url_for('dashboard'))

    users = ref.child('users').get() or {}
    forecasts = ref.child('forecasts').get() or {}

    total_users = len(users)
    active_users = sum(1 for user in users.values() if user.get('last_login'))
    total_predictions = sum(len(user_forecasts) for user_forecasts in forecasts.values())

    return render_template('admin.html',
                           users=users,
                           forecasts=forecasts,
                           user_data=user_data,
                           total_users=total_users,
                           active_users=active_users,
                           total_predictions=total_predictions)


@app.route('/admin/add_user', methods=['POST'])
def add_user():
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))

    email = request.form.get('email')
    password = request.form.get('password')
    display_name = request.form.get('display_name')
    role = request.form.get('role', 'user')

    try:
        user = auth.create_user(email=email, password=password)
        user_data = {
            'email': email,
            'display_name': display_name,
            'role': role,
            'created_at': datetime.now().isoformat()
        }
        ref.child(f'users/{user.uid}').set(user_data)
    except Exception:
        pass

    return redirect(url_for('admin'))


@app.route('/admin/delete_user/<uid>', methods=['POST'])
def delete_user(uid):
    if 'user' not in session:
        return redirect(url_for('login'))

    admin_uid = session['user']
    admin_data = ref.child(f"users/{admin_uid}").get() or {}

    if admin_data.get('role') != 'admin':
        return redirect(url_for('dashboard'))

    try:
        if uid == admin_uid:
            return redirect(url_for('admin'))

        ref.child(f"users/{uid}").delete()
        ref.child(f"forecasts/{uid}").delete()
        auth.delete_user(uid)
    except Exception:
        pass

    return redirect(url_for('admin'))


@app.route('/admin/clear_forecasts', methods=['POST'])
def clear_all_forecasts():
    if 'user' not in session:
        return redirect(url_for('login'))

    admin_uid = session['user']
    admin_data = ref.child(f"users/{admin_uid}").get() or {}

    if admin_data.get('role') != 'admin':
        return redirect(url_for('dashboard'))

    try:
        ref.child('forecasts').delete()
    except Exception:
        pass

    return redirect(url_for('admin'))


@app.route('/admin/edit_user/<uid>', methods=['GET', 'POST'])
def edit_user(uid):
    if 'user' not in session:
        return redirect(url_for('login'))

    admin_uid = session['user']
    admin_data = ref.child(f"users/{admin_uid}").get() or {}
    if admin_data.get('role') != 'admin':
        return redirect(url_for('dashboard'))

    user_data = ref.child(f"users/{uid}").get() or {}
    if not user_data:
        return redirect(url_for('admin'))

    if request.method == 'POST':
        try:
            email = request.form.get('email')
            display_name = request.form.get('display_name')
            company_name = request.form.get('company_name')
            role = request.form.get('role')

            if email != user_data.get('email'):
                auth.update_user(uid, email=email)

            updates = {
                'email': email,
                'role': role
            }

            if display_name:
                updates['display_name'] = display_name
            if company_name:
                updates['company_name'] = company_name

            ref.child(f"users/{uid}").update(updates)
            return redirect(url_for('admin'))

        except Exception:
            return redirect(url_for('admin'))

    return render_template('edit_user.html', user_data=user_data, uid=uid)


@app.route('/api/v1/forecasts', methods=['GET'])
def get_forecasts():
    api_key = request.headers.get('Authorization')

    if not api_key:
        return jsonify({'error': 'No API key provided'}), 401

    api_key = api_key.replace('Bearer ', '')
    users_ref = ref.child('users')

    user_with_key = None
    for uid, user_data in users_ref.get().items():
        if user_data.get('api_key') == api_key:
            user_with_key = uid
            break

    if not user_with_key:
        return jsonify({'error': 'Invalid API key'}), 401

    forecasts = ref.child(f"forecasts/{user_with_key}").get() or {}
    return jsonify(forecasts)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
