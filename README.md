## 📊 Sales Prediction System  
*A Flask-based machine learning web application for predicting sales revenue using product-related features. 
The system integrates Firebase for authentication and database operations, offers an admin dashboard, supports CSV/JSON exports, and is fully containerized for cloud deployment.*

---

## 🚀 Key Features

- 🔒 **User Authentication**  
  - Login/Signup via **Firebase Auth**  
  - **Role-based access** (Admin/User)

- 🤖 **Machine Learning Integration**  
  - Predicts sales revenue using a **Gradient Boosting** model  
  - Accepts **Product IDs** in `PXXXX` format (e.g., `P1001`)

- 📈 **Interactive Dashboard**  
  - Revenue trends & product category distribution via **Chart.js**  
  - Export prediction results to **CSV** or **JSON**

- 🛠️ **Admin Panel**  
  - Manage users and view system analytics

- 🐳 **Dockerized Deployment**  
  - Multi-stage Docker build for easy containerization

- ☁️ **Cloud Ready**  
  - Seamless deployment on **Azure Web Apps**

---

## 🧰 Tech Stack

### 🔙 Backend  
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)  
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask)  
![Firebase](https://img.shields.io/badge/Firebase-Realtime_DB-orange?logo=firebase)

### 🎨 Frontend  
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.0-purple?logo=bootstrap)  
![Chart.js](https://img.shields.io/badge/Chart.js-3.0-yellow?logo=chart.js)

### 🧠 Machine Learning  
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-green?logo=scikit-learn)

---

## 📦 Installation  
```bash
# Clone repo
git clone https://github.com/yourusername/sales-prediction.git
cd sales-prediction

# Install dependencies
pip install -r requirements.txt

# Train model (generates model.pkl)
python train_model.py

# Run app
python app.py

```

---
## 🚀 Deployment Options

### 🐳 Docker Deployment (Local)

```bash
# Build the Docker image
docker build -t sales-predictor .

# Run the container
docker run -d -p 5000:5000 --name sales-app sales-predictor

```
## ☁️ Azure Web App Deployment
```bash
1. Create the Azure Web App
az webapp create \
  --name sales-predictor-app \
  --resource-group sales-predictor-rg \
  --plan sales-predictor-plan \
  --runtime "PYTHON|3.9"

2. Configure the Docker Container

-Set the web app to use the Docker image from Docker Hub:

az webapp config container set \
  --name sales-predictor-app \
  --resource-group sales-predictor-rg \
  --container-image-name nipul274/sales-predictor:latest \
  --container-registry-url https://index.docker.io
```
## 📁 Project Structure: `sales-predictor`
```text
sales-predictor/
sales-prediction-web-app/
├── templates/               # HTML templates
├── static/                  # CSS/JS files
├── app.py                   # Flask backend
├── train_model.py           # ML model training script
├── requirements.txt         
├── Dockerfile               
├── .gitignore               
├── README.md                
├── model.pkl                # (ignored) trained ML model
├── model_columns.json       # (ignored) model feature metadata
├── sales_data.csv           # (ignored) training data
└── serviceAccountKey.json   # (ignored) Firebase credentials

```

