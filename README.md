<div align="center">
  <h1>🚀 SwiftRelief</h1>
  <p><b>Intelligent, AI-Powered Hospital Recommendation System</b></p>
  
  <img src="https://img.shields.io/badge/Flutter-02569B?style=for-the-badge&logo=flutter&logoColor=white" alt="Flutter" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/CatBoost-FFD94C?style=for-the-badge&logo=catboost&logoColor=black" alt="CatBoost" />
</div>

<br>

## 📖 Overview

**SwiftRelief** is a cross-platform system engineered to optimize hospital discovery and patient routing. By combining an intuitive mobile interface with a robust Machine Learning backend, the platform delivers instantaneous, location-aware hospital recommendations tailored to user needs, location, and medical urgency.

## ✨ Key Technical Highlights & Architecture

### 🧠 Advanced Machine Learning Engine
- **Contextual Ranking Systems**: Implements a highly tuned **CatBoost Ranker** algorithm to evaluate and prioritize hospitals and medical facilities based on multi-dimensional criteria (distance, specialty, user urgency, and facility capacity).
- **Deep Learning Subsystems**: Utilizes **TensorFlow** neural networks for structured data classification and multi-seed training pipelines.
- **Explainable AI (XAI)**: Includes specialized pipelines to interpret model decisions, ensuring healthcare recommendations are transparent, auditable, and unbiased.

### 🏗️ Resilient Systems Engineering
- **Offline-First Capabilities**: Geolocation and essential places data are defensively cached locally. The Flutter app is designed to gracefully degrade, serving cached hospital maps when cellular connectivity is severely limited.
- **Optimized Data Pipelines**: Features a robust synthetic data generator to simulate complex user queries and hospital resource scenarios, facilitating rigorous, data-driven unit testing and model evaluation without exposing sensitive healthcare datasets.

### 📱 Cross-Platform Frontend
- **Unified Codebase**: Built utilizing **Flutter** and **Dart**, compiling natively to Android, iOS, and Web for maximum accessibility.
- **Real-Time Integration**: Architectured for zero-latency interactions, communicating securely with the backend API for live location updates and dynamic resource allocation.

---

## 🔌 API Reference (Core Endpoints)

The Python Flask backend provides a robust RESTful API for the mobile clients. 

| Endpoint | Method | Description | Auth Required |
| :--- | :---: | :--- | :---: |
| `/api/auth/register` | `POST` | Register a new user and store essential medical background. | ❌ |
| `/api/auth/login` | `POST` | Authenticate user & receive JWT access token. | ❌ |
| `/api/profile` | `GET`/`PUT` | Retrieve or update user profile and chronic conditions. | 🔒 Yes |
| `/api/geocode` | `GET` | Convert string location queries into exact lat/lon coordinates. | 🔒 Yes |
| `/api/recommend` | `POST` | **Core ML Endpoint:** Returns ranked hospital recommendations. | 🔒 Yes |
| `/api/map_symptom` | `POST` | Maps raw user symptoms to medical specialties (LLM-assisted). | 🔒 Yes |
| `/api/feedback` | `POST` | Submit relevance feedback (thumbs up/down) for ML reinforcement. | 🔒 Yes |
| `/api/health` | `GET` | Check backend system, database, and inference engine status. | ❌ |

---

## 📐 System Architecture

```text
[ Mobile Clients ]  --> (RESTful API) --> [ Application Layer ]
  (Flutter/Dart)                          (Python / Flask)
       |                                         |
       v                                         v
 [ Local Cache ]                          [ ML Inference Engine ]
 (Offline-first /                         - TensorFlow Estimators
  Graceful Degradation)                   - CatBoost Ranking Models
                                                 |
                                                 v
                                  [ Intelligent Caching & Data Logic ]
                                  - Geocode caching
                                  - Live Place Validation & Enrichment
```

---

## 🚀 Getting Started

### Prerequisites
- [Flutter SDK](https://docs.flutter.dev/get-started/install) (v3.0+)
- Python 3.8+
- Virtual Environment tool (`venv` or `conda`)

### 1. Backend Services Setup
The backend exposes the ML models and data endpoints via a lightweight WSGI framework.

```bash
# Navigate to backend module
cd swiftrelief_backend

# Isolate environment (Windows example)
python -m venv .venv
.venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Boot up the server
python app.py
```

### 2. Mobile Client Setup

```bash
# Navigate to frontend module
cd swiftrelief_flutter

# Fetch dependencies
flutter pub get

# Run the application (ensure an emulator or physical device is connected)
flutter run
```

---

## 🛠️ Technology Stack

| Domain | Technologies |
| :--- | :--- |
| **Frontend** | Flutter, Dart, Material Design |
| **Backend API** | Python, Flask, RESTful Design |
| **Machine Learning** | CatBoost, TensorFlow, Scikit-Learn, Pandas, NumPy |
| **Mapping & Location** | Geolocation caching, Google Maps & Places integrations |

---

## 📈 Roadmap & Future Developments
- [ ] On-device ML execution on Android devices via **TensorFlow Lite** for zero-latency hospital ranking.
- [ ] Real-time WebSocket support for live hospital bed availability and status updates.
- [ ] Integration with hospital booking, appointment, and emergency triage systems.

