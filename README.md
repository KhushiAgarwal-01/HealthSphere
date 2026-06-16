## **AmarCare- AI-Powered Health Assistant**
 # HealthPredict Pro is an intelligent healthcare platform that combines machine learning disease prediction with Gemini AI-powered medical assistance. The system helps users assess their risk for diabetes, heart disease, and kidney disease while providing instant health information through an AI chatbot.



## 📋 Table of Contents
🌟 Overview
✨ Features
🏗️ System Architecture
⚙️ Installation & Setup
🚀 Usage Guide
🔧 Technical Details
📁 Project Structure
🤝 Contributing
📄 License
📞 Contact

## 🌟 Overview

HealthPredict Pro is an advanced healthcare application that combines machine learning disease prediction with Gemini AI-powered health assistance. The system provides:

- Disease Risk Assessment for Heart Disease, and Kidney Disease
- AI Health Chatbot with multimodal support (text, images, PDFs)
- Nearby Doctor Finder with geolocation
- Personalized Health Advice based on prediction results

## 🎯 Key Objectives

- Provide accessible health risk assessment
- Offer AI-powered medical information
- Connect users with healthcare resources
- Ensure data privacy and security
- Deliver user-friendly interface

## ✨ Features


# 🤖 AI Health Assistant

- Gemini AI Integration: Google's advanced AI model

# 📍 Smart Doctor Finder

- Geolocation Services: Finds doctors near user location
- Specialty Mapping: Connects diseases to relevant specialists
- Mock Data System: Demonstrates functionality
- Google Maps Integration: Get directions instantly
- Contact Information: Phone numbers and addresses

# 🎨 User Experience

- Responsive Design: Works on desktop and mobile
- Pop-up Chatbot: Accessible from any page
- Interactive Forms: User-friendly input interface
- Visual Feedback: Animations and loading indicators



## 🏗️ System Architecture


``` mermaid
graph TB
    A[User Access] --> B{Home Page}
    B --> C[Disease Prediction]
    B --> D[AI Chatbot]
    B --> E[Health Resources]
    

    C --> C2[Heart Check]
    C --> C3[Kidney Check]
    
    C1 --> F[Input Health Parameters]
    C2 --> F
    C3 --> F
    
    F --> G[ML Model Processing]
    G --> H[Prediction Result]
    H --> I[Health Advice]
    I --> J[Find Doctors]
    
    D --> K[Ask Health Questions]
    D --> L[Upload Documents]
    K --> M[Gemini AI Processing]
    L --> M
    M --> N[AI Response]
    
    J --> O[Geolocation]
    O --> P[Doctor Results]
    P --> Q[Navigation & Contact]
    
    subgraph "Backend Services"
        R[Flask Server]
        S[Machine Learning Models]
        T[Gemini AI API]
        U[Database]
    end
    
    G --> R
    M --> T
    J --> R
    R --> S
    R --> T
    R --> U
```

## ⚙️ Installation & Setup

# 📋 Prerequisites
 - Python 3.8 or higher
 - Google Gemini API Key
 - Modern web browser
 - Git (for version control)

## 🚀 Quick Installation
```
# 1. Clone the repository
git clone https://github.com/yourusername/healthpredict-pro.git
cd healthpredict-pro

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set up environment variables
cp .env.example .env
# Edit .env file with your Gemini API key

# 6. Run the application
python app.py
```
## 🔑 Environment Configuration

```
# Flask Configuration
SECRET_KEY=your-secret-key-here
DEBUG=True

# Gemini API Configuration
GEMINI_API_KEY=your-google-gemini-api-key-here

# Application Settings
UPLOAD_FOLDER=./uploads
MAX_CONTENT_LENGTH=16777216  # 16MB
```
## 📦 Dependencies

```
Flask==2.3.2
google-generativeai==0.3.0
numpy==1.24.3
scikit-learn==1.3.0
Pillow==10.0.0
PyPDF2==3.0.0
python-dotenv==1.0.0
flask-cors==4.0.0
```
## 🔧 Technical Details
🧠 Machine Learning Models

Model Specifications:

Disease	Algorithm	| Accuracy          |	Dataset	Features
Heart Disease	    | SVM	            |     87%	Cleveland	13 Clinical Parameters
Kidney Disease	    | Gradient Boosting |	  90%	CKD Dataset	24 Medical Attributes

## 📁 Project Structure

```
healthpredict-pro/
├── 📁 templates/           # HTML Templates
│   ├── base.html          # Main template
│   ├── home.html          # Home page
│   ├── heart.html         # Heart disease prediction
│   ├── kidney.html        # Kidney disease prediction
│   ├── about.html         # About page
│   ├── 404.html           # Error page
│   └── 500.html           # Server error page
│
├── 📁 static/             # Static Assets
│   ├── 📁 css/
│   │   └── style.css     # Custom styles
│   ├── 📁 js/
│   │   ├── main.js       # Main JavaScript
│   │   └── chatbot.js    # Chatbot functionality
│   └── 📁 images/        # Image assets
│
├── 📁 saved_models/       # ML Models
│   ├── heart.pkl         # Heart disease model
│   └── kidney.pkl        # Kidney disease model
│
├── 📁 uploads/           # Temporary file storage
│   └── .gitkeep         # Keep folder in git
│
├── app.py               # Main Flask application
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── .env.example       # Environment template
├── .gitignore         # Git ignore file
├── README.md          # This file
└── LICENSE            # MIT License

<div align="center">
🌟 Made with ❤️ for Better Healthcare
Empowering Health Decisions Through AI & Machine Learning

</div>

