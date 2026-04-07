from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, send_file, session
from flask_cors import CORS  # Add this import
import pickle
import os
import numpy as np
from datetime import datetime
import random
# import google.generativeai as genai
from werkzeug.utils import secure_filename
import PyPDF2
from PIL import Image
import io
import base64
import json
from dotenv import load_dotenv
from google import genai
# import google.generativeai as genai 
from ecg_predict import load_ecg_models, predict_ecg
# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Add CORS support
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Configuration
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', './uploads')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16777216))  # 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf', 'txt'}
app.config['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure Gemini API
CHATBOT_NAME = "HealthAI Assistant"

# Initialize the client
gemini_client = None

try:
    if app.config['GEMINI_API_KEY']:
        # Initialize with the NEW SDK
        gemini_client = genai.Client(api_key=app.config['GEMINI_API_KEY'])
        
        print("✅ Gemini API configured successfully with NEW SDK")
        print(f"✅ API Key: {app.config['GEMINI_API_KEY'][:10]}...")
        
    else:
        print("❌ Warning: GEMINI_API_KEY not found in environment variables")
except Exception as e:
    print(f"❌ Error configuring Gemini API: {str(e)}")
    import traceback
    traceback.print_exc()

# Load ML models
working_dir = os.path.dirname(os.path.abspath(__file__))
models = {}

try:
    models['diabetes'] = pickle.load(open(f'{working_dir}/saved_models/diabetes.pkl', 'rb'))
    models['heart'] = pickle.load(open(f'{working_dir}/saved_models/heart.pkl', 'rb'))
    models['kidney'] = pickle.load(open(f'{working_dir}/saved_models/kidney.pkl', 'rb'))
    print("✓ ML models loaded successfully")
except Exception as e:
    print(f"✗ Error loading models: {str(e)}")
    # Create dummy models for development if real ones fail
    class DummyModel:
        def predict(self, X):
            return [random.randint(0, 1)]
        def predict_proba(self, X):
            return [[random.random(), random.random()]]
    
    for model_name in ['diabetes', 'heart', 'kidney']:
        models[model_name] = DummyModel()

try:
    load_ecg_models()
    print("✓ ECG models loaded successfully")
except Exception as e:
    print(f"✗ Error loading ECG models: {str(e)}")

# Utility Functions
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(file_path):
    """Extract text content from PDF files"""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num, page in enumerate(reader.pages):
                if page_num >= 5:  # Limit to first 5 pages
                    text += "\n[Document truncated after 5 pages]"
                    break
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text[:10000]  # Limit to 10,000 characters
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def get_gemini_response(user_message, file_path=None, file_type=None):
    """Get response from Gemini API with NEW SDK syntax"""
    
    if not gemini_client:
        return "⚠️ Health information service is currently unavailable. Please try again later."

    # Enhanced health-specific system prompt
    system_prompt = """You are HealthAI Assistant, a specialized healthcare AI with expertise in:
    1. Medical information and disease education
    2. Symptom analysis (NOT diagnosis)
    3. Health and wellness guidance
    4. Medication information
    5. Nutrition and exercise advice
    6. Mental health support
    7. Medical document/image analysis (lab results, prescriptions, etc.)

    CRITICAL RULES YOU MUST FOLLOW:
    1. NEVER provide medical diagnoses - always recommend consulting healthcare professionals
    2. For emergency symptoms (chest pain, difficulty breathing, severe bleeding), always advise immediate medical attention
    3. Be empathetic, accurate, and professional
    4. If analyzing medical documents, focus on explaining terminology, not providing interpretations
    5. Always include a disclaimer that you are not a medical professional
    6. If unsure about something, admit your limitations
    7. Never recommend specific medications or dosages
    8. Always encourage follow-up with healthcare providers
    
    Format responses with:
    - Clear headings for different sections
    - Bullet points for lists
    - Bold text for important warnings
    - A clear disclaimer at the end
    
    Tone: Professional, empathetic, helpful but cautious."""

    try:
        if file_path and os.path.exists(file_path):
            if file_type and file_type.startswith('image/'):
                # Image analysis with Gemini Vision
                try:
                    from PIL import Image
                    img = Image.open(file_path)
                    
                    # Prepare the prompt for image analysis
                    full_prompt = f"{system_prompt}\n\nUser's message: {user_message}\n\nPlease analyze this image for health-related content. Remember: Do not diagnose, only provide educational information about what you see."
                    
                    # NEW SDK syntax for image
                    response = gemini_client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=[full_prompt, img]
                    )
                    
                    # Add additional disclaimer for image analysis
                    result = response.text + "\n\n**⚠️ Important**: This image analysis is for educational purposes only and should not be used for diagnosis. Please consult a healthcare professional for medical advice."
                    return result
                    
                except Exception as img_error:
                    print(f"Image analysis error: {img_error}")
                    return "⚠️ I had trouble analyzing the image. Please make sure it's a clear image and try again, or consult a healthcare professional directly."
            
            elif file_type == 'application/pdf':
                # PDF text analysis
                try:
                    pdf_text = extract_text_from_pdf(file_path)
                    
                    if "Error reading PDF" in pdf_text:
                        return "⚠️ I couldn't read the PDF file properly. Please make sure it's not password protected and contains extractable text."
                    
                    # Truncate if too long for context
                    if len(pdf_text) > 8000:
                        pdf_text = pdf_text[:8000] + "\n\n[Document truncated due to length]"
                    
                    prompt = f"""{system_prompt}

                    DOCUMENT CONTENT (for context only):
                    {pdf_text}

                    USER'S QUESTION:
                    {user_message}

                    INSTRUCTIONS:
                    1. Explain medical terminology found in the document
                    2. Do NOT interpret results or provide diagnoses
                    3. Suggest what type of healthcare professional to consult
                    4. Include important disclaimers"""
                    
                    # NEW SDK syntax for text
                    response = gemini_client.models.generate_content(
                        model="gemini-1.5-pro",
                        contents=prompt
                    )
                    return response.text
                    
                except Exception as pdf_error:
                    print(f"PDF analysis error: {pdf_error}")
                    return "⚠️ I had trouble analyzing the PDF. Please try again or consult a healthcare professional for interpretation of medical documents."
        
        # Text-only request
        prompt = f"""{system_prompt}
        
        USER'S QUESTION:
        {user_message}
        
        Please provide a helpful, informative response following all the rules above."""
        
        # Use gemini-2.5-flash for chat (NEW model)
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
        
    except Exception as e:
        print(f"🔥 Gemini API error: {str(e)}")
        print(f"🔥 Error type: {type(e).__name__}")
        
        # Provide helpful error messages
        if "API_KEY_INVALID" in str(e) or "403" in str(e):
            return "⚠️ Invalid Gemini API key. Please check your .env file configuration."
        elif "429" in str(e) or "quota" in str(e).lower():
            return "⚠️ API quota exceeded. Please try again later or check your Google Cloud billing."
        elif "503" in str(e) or "unavailable" in str(e).lower():
            return "⚠️ Gemini API service temporarily unavailable. Please try again in a moment."
        elif "model not found" in str(e).lower():
            return "⚠️ Model not found. Please use 'gemini-2.0-flash' or 'gemini-1.5-flash' instead."
        else:
            return f"⚠️ I'm experiencing difficulties connecting to the health information service. Error: {str(e)[:100]}"

# ADDING NEW DIET RECOMMENDATION-----------------------------------------------------------
def generate_diet_recommendation(age, cholesterol, bp, bmi, prediction):

    if not gemini_client:
        return "AI diet recommendation service unavailable."

    try:
        prompt = f"""
        A patient has the following health information:

        Age: {age}
        Cholesterol Level: {cholesterol}
        Blood Pressure: {bp}
        BMI: {bmi}
        Heart Disease Risk Prediction: {prediction}

        Provide personalized heart-healthy diet recommendations.

        Include:
        - Foods to eat
        - Foods to avoid
        - Lifestyle suggestions

        Keep the answer short with bullet points.
        """

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        print("Diet AI error:", e)
        return "Unable to generate personalized diet recommendation."


# Health Advice Function
def get_health_advice(disease, has_disease):
    advice = {
        "general_tips": [],
        "doctor_visit": "",
        "resources": []
    }
    
    if disease == "diabetes":
        if has_disease:
            advice["general_tips"] = [
                "Monitor your blood sugar levels regularly",
                "Follow a balanced diet low in simple carbohydrates",
                "Engage in regular physical activity",
                "Take prescribed medications as directed"
            ]
            advice["doctor_visit"] = "Schedule an appointment with an endocrinologist or your primary care physician"
            advice["resources"] = [
                "American Diabetes Association: www.diabetes.org",
                "National Institute of Diabetes and Digestive and Kidney Diseases: www.niddk.nih.gov"
            ]
        else:
            advice["general_tips"] = [
                "Maintain a healthy weight",
                "Exercise regularly (at least 150 minutes per week)",
                "Limit sugar and refined carbohydrate intake",
                "Get regular health checkups"
            ]
    
    elif disease == "heart":
        if has_disease:
            advice["general_tips"] = [
                "Follow a heart-healthy diet (Mediterranean diet recommended)",
                "Quit smoking if you currently smoke",
                "Manage stress through relaxation techniques",
                "Take all prescribed medications regularly"
            ]
            advice["doctor_visit"] = "Schedule an appointment with a cardiologist immediately"
            advice["resources"] = [
                "American Heart Association: www.heart.org",
                "Cardiology department at your nearest hospital"
            ]
        else:
            advice["general_tips"] = [
                "Maintain healthy blood pressure and cholesterol levels",
                "Exercise for at least 30 minutes most days",
                "Eat a diet rich in fruits, vegetables, and whole grains",
                "Avoid tobacco products"
            ]
    
    elif disease == "kidney":
        if has_disease:
            advice["general_tips"] = [
                "Monitor blood pressure regularly",
                "Reduce sodium intake",
                "Stay hydrated with water",
                "Avoid NSAIDs (like ibuprofen) unless prescribed"
            ]
            advice["doctor_visit"] = "Schedule an appointment with a nephrologist as soon as possible"
            advice["resources"] = [
                "National Kidney Foundation: www.kidney.org",
                "Your local nephrology center"
            ]
        else:
            advice["general_tips"] = [
                "Drink plenty of water",
                "Maintain healthy blood pressure",
                "Limit salt and processed foods",
                "Get regular kidney function tests if at risk"
            ]
    
    return advice

# ===================== CHATBOT ROUTES =====================

@app.route('/chatbot_interface')
def chatbot_interface():
    """Creative chatbot interface with multimodal support"""
    return render_template('chatbot.html', chatbot_name=CHATBOT_NAME)

@app.route('/api/chatbot-test')
def chatbot_test():
    """Test page for chatbot"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chatbot Test</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            .chat { border: 1px solid #ccc; padding: 20px; height: 300px; overflow-y: scroll; margin-bottom: 20px; }
            .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
            .user { background: #007bff; color: white; text-align: right; }
            .bot { background: #f1f1f1; }
        </style>
    </head>
    <body>
        <h1>Chatbot Connection Test</h1>
        <div id="chat" class="chat"></div>
        <input type="text" id="message" placeholder="Type your message..." style="width: 70%; padding: 10px;">
        <button onclick="sendMessage()" style="padding: 10px 20px;">Send</button>
        
        <script>
            async function sendMessage() {
                const input = document.getElementById('message');
                const message = input.value;
                if (!message) return;
                
                // Add user message
                addMessage(message, 'user');
                input.value = '';
                
                try {
                    const response = await fetch('/chatbot', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ message: message })
                    });
                    
                    const data = await response.json();
                    addMessage(data.response, 'bot');
                    
                } catch (error) {
                    addMessage('Error: ' + error.message, 'bot');
                }
            }

            # async function sendMessage() {

            #     const messageInput = document.getElementById("message");
            #     const message = messageInput.value.trim();

            #     if (!message) return;

            #     const formData = new FormData();
            #     formData.append("message", message);

            #     try {

            #         const response = await fetch("/chatbot", {
            #             method: "POST",
            #             body: formData
            #         });

            #         const data = await response.json();
            #         addMessage(data.response, "bot");

            #     } catch (error) {
            #         addMessage("Error: " + error.message, "bot");
            #     }

            # }
                        
            function addMessage(text, sender) {
                const chat = document.getElementById('chat');
                const div = document.createElement('div');
                div.className = 'message ' + sender;
                div.textContent = text;
                chat.appendChild(div);
                chat.scrollTop = chat.scrollHeight;
            }
            
            // Test on Enter key
            document.getElementById('message').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') sendMessage();
            });
        </script>
    </body>
    </html>
    '''

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """Enhanced chatbot endpoint with multimodal support - FIXED VERSION"""
    try:
        print(f"=== Chatbot Request Received ===")
        print(f"Content-Type: {request.content_type}")
        print(f"Method: {request.method}")
        print(f"Has files: {'file' in request.files}")
        print(f"Form data: {request.form}")
        
        user_message = ""
        file = None
        file_path = None
        file_type = None
        
        # Check content type and handle accordingly
        if request.content_type and 'application/json' in request.content_type:
            # Handle JSON request
            print("Processing JSON request")
            data = request.get_json()
            if not data:
                return jsonify({
                    'response': 'Please provide a message in JSON format.',
                    'type': 'error',
                    'has_file': False
                }), 400
            
            user_message = data.get('message', '').strip()
            print(f"JSON message: {user_message}")
            
        elif request.content_type and 'multipart/form-data' in request.content_type:
            # Handle form data with file upload
            print("Processing form data with file upload")
            user_message = request.form.get('message', '').strip()
            file = request.files.get('file')
            print(f"Form message: {user_message}")
            print(f"File received: {file.filename if file else 'None'}")
            
        else:
            # Try to get message from form data
            print("Trying form data")
            user_message = request.form.get('message', '').strip()
            file = request.files.get('file')
        
        # Validate input
        if not user_message and not file:
            print("No message or file provided")
            return jsonify({
                'response': 'Please enter a message or upload a file.',
                'type': 'error',
                'has_file': False
            }), 400
        
        # Process uploaded file if exists
        if file and file.filename:
            print(f"Processing file: {file.filename}")
            if not allowed_file(file.filename):
                return jsonify({
                    'response': 'File type not allowed. Please upload PNG, JPG, JPEG, or PDF files only.',
                    'type': 'error',
                    'has_file': False
                }), 400
            
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_type = file.content_type
            
            # Validate file size
            file_size = os.path.getsize(file_path)
            if file_size > app.config['MAX_CONTENT_LENGTH']:
                os.remove(file_path)
                return jsonify({
                    'response': f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] // 1048576}MB.',
                    'type': 'error',
                    'has_file': False
                }), 400
        
        print(f"Getting response from Gemini for: {user_message[:50]}...")
        # Get response from Gemini
        bot_response = get_gemini_response(user_message, file_path, file_type)
        print(f"Got response from Gemini: {bot_response[:50]}...")
        
        # Clean up uploaded file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Cleaned up file: {file_path}")
            except Exception as e:
                print(f"Error cleaning up file: {e}")
        
        response_data = {
            'response': bot_response,
            'type': 'success',
            'has_file': bool(file)
        }
        
        print(f"Sending response: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"!!! Chatbot error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'response': f'An error occurred: {str(e)}',
            'type': 'error',
            'has_file': False
        }), 500



# DIET RECOMMENDATION ROUTE----------------------

@app.route('/diet', methods=['GET','POST'])
def diet():

    # Check if heart prediction data exists
    heart_data = session.get("heart_data")

    if heart_data:

        diet_recommendation = generate_diet_recommendation(
            heart_data["age"],
            heart_data["cholesterol"],
            heart_data["blood_pressure"],
            heart_data["bmi"],
            heart_data["prediction"]
        )

        return render_template(
        "diet.html",
        diet_recommendation=diet_recommendation,
        show_result=True,
        auto_generated=True,
        age=heart_data["age"],
        cholesterol=heart_data["cholesterol"],
        blood_pressure=heart_data["blood_pressure"],
        bmi=heart_data["bmi"]
    )

    # If user manually fills form
    if request.method == "POST":

        age = float(request.form.get('age'))
        cholesterol = float(request.form.get('cholesterol'))
        blood_pressure = float(request.form.get('blood_pressure'))
        bmi = float(request.form.get('bmi'))

        diet_recommendation = generate_diet_recommendation(
            age,
            cholesterol,
            blood_pressure,
            bmi,
            "Unknown"
        )
        session.pop('heart_data', None)

        return render_template(
        "diet.html",
        show_result=False,
        age="",
        cholesterol="",
        blood_pressure="",
        bmi=""
    )

    return render_template("diet.html", show_result=False)



# ===================== DOCTOR FINDER API =====================

@app.route('/api/nearby-doctors', methods=['POST'])
def nearby_doctors():
    """Find nearby doctors (mock implementation)"""
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        specialty = data.get('specialty', 'general')
        lat = data.get('latitude', 0)
        lng = data.get('longitude', 0)
        
        specialty_map = {
            'diabetes': 'Endocrinologist',
            'heart': 'Cardiologist',
            'kidney': 'Nephrologist',
            'general': 'General Practitioner'
        }
        
        specialty_name = specialty_map.get(specialty, 'Specialist')
        
        # Generate mock doctor data
        doctors = []
        for i in range(1, 4):
            doctors.append({
                'id': i,
                'name': f"{specialty_name} Center #{i}" if i > 1 else f"City General Hospital - {specialty_name}",
                'distance': round(random.uniform(0.5, 5.0), 1),
                'phone': f"(555) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
                'opening_hours': f"Open until {random.randint(5, 8)}:{'00' if random.random() > 0.5 else '30'} PM",
                'rating': round(random.uniform(3.5, 5.0), 1),
                'review_count': random.randint(50, 200),
                'address': f"{random.randint(100, 999)} {'Main' if i == 1 else 'Medical'} St.",
                'map_link': f"https://www.google.com/maps?q={specialty_name.replace(' ', '+')}+near+me@{lat},{lng}",
                'image_url': f"https://source.unsplash.com/random/300x200/?hospital,{specialty}"
            })
        
        return jsonify({
            'success': True,
            'specialtyName': specialty_name,
            'results': doctors
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ===================== MAIN ROUTES =====================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/emergency_info')
def emergency_info():
    """Emergency information page"""
    emergency_contacts = {
        "USA": {
            "Emergency": "911",
            "Suicide Prevention": "988",
            "Poison Control": "1-800-222-1222",
            "Domestic Violence": "1-800-799-7233"
        },
        "UK": {
            "Emergency": "999 or 112",
            "NHS Non-emergency": "111",
            "Samaritans (Suicide Prevention)": "116 123"
        },
        "Canada": {
            "Emergency": "911",
            "Suicide Prevention": "1-833-456-4566",
            "Poison Control": "1-844-764-7669"
        }
    }
    
    emergency_symptoms = [
        "Chest pain or pressure (especially radiating to arm, jaw, or back)",
        "Difficulty breathing or shortness of breath",
        "Severe bleeding that won't stop",
        "Sudden weakness or numbness in face, arm, or leg (especially on one side)",
        "Sudden confusion, trouble speaking, or understanding",
        "Sudden trouble seeing in one or both eyes",
        "Sudden severe headache with no known cause",
        "Fainting or unconsciousness",
        "Suicidal or homicidal thoughts",
        "Severe burns",
        "Choking",
        "Seizures that last more than 5 minutes",
        "Severe allergic reaction (difficulty breathing, swelling of face/throat)"
    ]
    
    return render_template('emergency.html',
                         emergency_contacts=emergency_contacts,
                         emergency_symptoms=emergency_symptoms)

@app.route('/health_tips')
def health_tips():
    """General health tips page"""
    tips_by_category = {
        "Nutrition": [
            "Eat a variety of colorful fruits and vegetables daily",
            "Choose whole grains over refined grains",
            "Limit added sugars and saturated fats",
            "Stay hydrated with water instead of sugary drinks",
            "Practice portion control"
        ],
        "Exercise": [
            "Aim for at least 150 minutes of moderate exercise per week",
            "Include strength training twice a week",
            "Take breaks from sitting every 30 minutes",
            "Find activities you enjoy to stay consistent",
            "Warm up before and cool down after exercise"
        ],
        "Mental Health": [
            "Practice mindfulness or meditation daily",
            "Maintain social connections",
            "Get 7-9 hours of quality sleep per night",
            "Set realistic goals and celebrate small wins",
            "Seek professional help when needed"
        ],
        "Preventive Care": [
            "Get regular health check-ups",
            "Stay up to date on vaccinations",
            "Know your family medical history",
            "Don't ignore persistent symptoms",
            "Follow screening guidelines for your age group"
        ]
    }
    
    return render_template('health_tips.html', tips_by_category=tips_by_category)

# ===================== DISEASE PREDICTION ROUTES =====================

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        try:
            pregnancies = float(request.form.get('pregnancies', 0))
            glucose = float(request.form.get('glucose', 0))
            blood_pressure = float(request.form.get('blood_pressure', 0))
            skin_thickness = float(request.form.get('skin_thickness', 0))
            insulin = float(request.form.get('insulin', 0))
            bmi = float(request.form.get('bmi', 0))
            diabetes_pedigree = float(request.form.get('diabetes_pedigree', 0))
            age = float(request.form.get('age', 0))
            
            if not (0 <= pregnancies <= 20) or not (0 <= age <= 120):
                flash("Please enter valid values for pregnancies (0-20) and age (0-120)", 'warning')
                return redirect(url_for('diabetes'))
            
            # BMI categories
            new_bmi_underweight = 1 if bmi <= 18.5 else 0
            new_bmi_overweight = 1 if 24.9 < bmi <= 29.9 else 0
            new_bmi_obesity_1 = 1 if 29.9 < bmi <= 34.9 else 0
            new_bmi_obesity_2 = 1 if 34.9 < bmi <= 39.9 else 0
            new_bmi_obesity_3 = 1 if bmi > 39.9 else 0
            
            # Insulin categories
            new_insulin_normal = 1 if 16 <= insulin <= 166 else 0
            
            # Glucose categories
            if glucose <= 70:
                new_glucose_low, new_glucose_normal, new_glucose_overweight, new_glucose_secret = 1, 0, 0, 0
            elif 70 < glucose <= 99:
                new_glucose_low, new_glucose_normal, new_glucose_overweight, new_glucose_secret = 0, 1, 0, 0
            elif 99 < glucose <= 126:
                new_glucose_low, new_glucose_normal, new_glucose_overweight, new_glucose_secret = 0, 0, 1, 0
            else:
                new_glucose_low, new_glucose_normal, new_glucose_overweight, new_glucose_secret = 0, 0, 0, 1
            
            user_input = [
                pregnancies, glucose, blood_pressure, skin_thickness, insulin,
                bmi, diabetes_pedigree, age, new_bmi_underweight,
                new_bmi_overweight, new_bmi_obesity_1, new_bmi_obesity_2,
                new_bmi_obesity_3, new_insulin_normal, new_glucose_low,
                new_glucose_normal, new_glucose_overweight, new_glucose_secret
            ]
            
            prediction = models['diabetes'].predict([user_input])
            result = "The person is predicted to have diabetes" if prediction[0] == 1 else "The person is predicted to not have diabetes"
            
            probability = models['diabetes'].predict_proba([user_input]) if hasattr(models['diabetes'], 'predict_proba') else [[0, 0]]
            confidence = round(np.max(probability) * 100, 2)
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            health_advice = get_health_advice("diabetes", prediction[0] == 1)
            
            return render_template('diabetes.html', 
                                result=result, 
                                confidence=confidence,
                                current_time=current_time,
                                show_result=True,
                                health_advice=health_advice,
                                disease_type="diabetes")
        
        except ValueError as ve:
            flash(f"Invalid input: {str(ve)}. Please enter numeric values.", 'danger')
            return redirect(url_for('diabetes'))
        except Exception as e:
            flash(f"An error occurred: {str(e)}", 'danger')
            return redirect(url_for('diabetes'))
    
    return render_template('diabetes.html', show_result=False)

@app.route('/heart', methods=['GET', 'POST'])
def heart():
    if request.method == 'POST':
        try:
            age = float(request.form.get('age', 0))
            sex = float(request.form.get('sex', 0))
            cp = float(request.form.get('cp', 0))
            trestbps = float(request.form.get('trestbps', 0))
            chol = float(request.form.get('chol', 0))
            fbs = float(request.form.get('fbs', 0))
            restecg = float(request.form.get('restecg', 0))
            thalach = float(request.form.get('thalach', 0))
            exang = float(request.form.get('exang', 0))
            oldpeak = float(request.form.get('oldpeak', 0))
            slope = float(request.form.get('slope', 0))
            ca = float(request.form.get('ca', 0))
            thal = float(request.form.get('thal', 0))
            
            if not (0 <= cp <= 3) or not (0 <= fbs <= 1) or not (0 <= restecg <= 2):
                flash("Please enter valid values for all fields", 'warning')
                return redirect(url_for('heart'))
            
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
                         exang, oldpeak, slope, ca, thal]
            
            prediction = models['heart'].predict([user_input])
            result = "This person is predicted to have heart disease" if prediction[0] == 1 else "This person is predicted to not have heart disease"
            session['heart_data'] = {
                "age": age,
                "cholesterol": chol,
                "blood_pressure": trestbps,
                "bmi": 25,
                "prediction": prediction[0]
            }
            probability = models['heart'].predict_proba([user_input]) if hasattr(models['heart'], 'predict_proba') else [[0, 0]]
            confidence = round(np.max(probability) * 100, 2)


            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            health_advice = get_health_advice("heart", prediction[0] == 1)

            # NEW: Generate AI diet recommendation
            diet_recommendation = generate_diet_recommendation(
                age,
                chol,
                trestbps,
                25,   # BMI placeholder (since heart dataset doesn't include BMI)
                prediction[0]
            )
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            health_advice = get_health_advice("heart", prediction[0] == 1)
            
            # return render_template('heart.html', 
            #                      result=result, 
            #                      confidence=confidence,
            #                      current_time=current_time,
            #                      show_result=True,
            #                      health_advice=health_advice,
            #                      disease_type="heart")

            return render_template('heart.html', 
                     result=result, 
                     confidence=confidence,
                     current_time=current_time,
                     show_result=True,
                     health_advice=health_advice,
                     diet_recommendation=diet_recommendation,
                     disease_type="heart")
        
        except ValueError as ve:
            flash(f"Invalid input: {str(ve)}. Please enter numeric values.", 'danger')
            return redirect(url_for('heart'))
        except Exception as e:
            flash(f"An error occurred: {str(e)}", 'danger')
            return redirect(url_for('heart'))
    
    return render_template('heart.html', show_result=False)


@app.route('/ecg', methods=['GET'])
def ecg():
    return render_template('ecg.html')


@app.route('/predict-ecg', methods=['POST'])
def predict_ecg_route():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['file']
        if not file or not file.filename:
            return jsonify({'error': 'No image selected'}), 400

        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if ext not in {'png', 'jpg', 'jpeg'}:
            return jsonify({'error': 'Invalid file type. Use PNG, JPG or JPEG.'}), 400

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        stored_name = f"ecg_{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_name)
        file.save(file_path)

        try:
            prediction = predict_ecg(file_path)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

        return jsonify({'prediction': prediction})

    except Exception as e:
        err = str(e)
        if "Repo ECG model could not be loaded in this Python/sklearn environment" in err:
            err = (
                "Repo ECG model is selected, but it cannot be loaded in the current Python/sklearn version. "
                "Please run this project in a repo-compatible environment (older sklearn) to use that model."
            )
        return jsonify({'error': f'Prediction failed: {err}'}), 500

@app.route('/kidney', methods=['GET', 'POST'])
def kidney():
    if request.method == 'POST':
        try:
            age = float(request.form.get('age', 0))
            blood_pressure = float(request.form.get('blood_pressure', 0))
            specific_gravity = float(request.form.get('specific_gravity', 0))
            albumin = float(request.form.get('albumin', 0))
            sugar = float(request.form.get('sugar', 0))
            red_blood_cells = float(request.form.get('red_blood_cells', 0))
            pus_cell = float(request.form.get('pus_cell', 0))
            pus_cell_clumps = float(request.form.get('pus_cell_clumps', 0))
            bacteria = float(request.form.get('bacteria', 0))
            blood_glucose_random = float(request.form.get('blood_glucose_random', 0))
            blood_urea = float(request.form.get('blood_urea', 0))
            serum_creatinine = float(request.form.get('serum_creatinine', 0))
            sodium = float(request.form.get('sodium', 0))
            potassium = float(request.form.get('potassium', 0))
            haemoglobin = float(request.form.get('haemoglobin', 0))
            packed_cell_volume = float(request.form.get('packed_cell_volume', 0))
            white_blood_cell_count = float(request.form.get('white_blood_cell_count', 0))
            red_blood_cell_count = float(request.form.get('red_blood_cell_count', 0))
            hypertension = float(request.form.get('hypertension', 0))
            diabetes_mellitus = float(request.form.get('diabetes_mellitus', 0))
            coronary_artery_disease = float(request.form.get('coronary_artery_disease', 0))
            appetite = float(request.form.get('appetite', 0))
            peda_edema = float(request.form.get('peda_edema', 0))
            aanemia = float(request.form.get('aanemia', 0))
            
            if not all(x in (0, 1) for x in [hypertension, diabetes_mellitus, coronary_artery_disease, appetite, peda_edema, aanemia]):
                flash("Please enter valid values (0 or 1) for binary fields", 'warning')
                return redirect(url_for('kidney'))
            
            user_input = [
                age, blood_pressure, specific_gravity, albumin, sugar,
                red_blood_cells, pus_cell, pus_cell_clumps, bacteria,
                blood_glucose_random, blood_urea, serum_creatinine, sodium,
                potassium, haemoglobin, packed_cell_volume,
                white_blood_cell_count, red_blood_cell_count, hypertension,
                diabetes_mellitus, coronary_artery_disease, appetite,
                peda_edema, aanemia
            ]
            
            prediction = models['kidney'].predict([user_input])
            result = "The person is predicted to have kidney disease" if prediction[0] == 1 else "The person is predicted to not have kidney disease"
            
            probability = models['kidney'].predict_proba([user_input]) if hasattr(models['kidney'], 'predict_proba') else [[0, 0]]
            confidence = round(np.max(probability) * 100, 2)
            
            current_time = datetime.now().strftime("%Y-%m-d %H:%M:%S")
            health_advice = get_health_advice("kidney", prediction[0] == 1)
            
            return render_template('kidney.html', 
                                result=result, 
                                confidence=confidence,
                                current_time=current_time,
                                show_result=True,
                                health_advice=health_advice,
                                disease_type="kidney")
        
        except ValueError as ve:
            flash(f"Invalid input: {str(ve)}. Please enter numeric values.", 'danger')
            return redirect(url_for('kidney'))
        except Exception as e:
            flash(f"An error occurred: {str(e)}", 'danger')
            return redirect(url_for('kidney'))
    
    return render_template('kidney.html', show_result=False)

# ===================== UTILITY ROUTES =====================

# @app.route('/api/status')
# def api_status():
#     """Check API and model status"""
#     status = {
#         "flask_app": "running",
#         "gemini_api": "configured" if gemini_models['chat'] else "not_configured",
#         "ml_models": {
#             "diabetes": "loaded" if 'diabetes' in models else "error",
#             "heart": "loaded" if 'heart' in models else "error",
#             "kidney": "loaded" if 'kidney' in models else "error"
#         },
#         "upload_folder": "exists" if os.path.exists(app.config['UPLOAD_FOLDER']) else "missing",
#         "timestamp": datetime.now().isoformat()
#     }
#     return jsonify(status)

# ===================== ERROR HANDLERS =====================

@app.errorhandler(404)
def page_not_found(e):
    try:
        return render_template('404.html'), 404
    except:
        return '''
        <!DOCTYPE html>
        <html>
        <head><title>404 Not Found</title></head>
        <body>
            <h1>404 - Page Not Found</h1>
            <p>The requested URL was not found on the server.</p>
            <a href="/">Go to Homepage</a>
        </body>
        </html>
        ''', 404

@app.errorhandler(500)
def internal_server_error(e):
    try:
        return render_template('500.html'), 500
    except:
        return '''
        <!DOCTYPE html>
        <html>
        <head><title>500 Internal Server Error</title></head>
        <body>
            <h1>500 - Internal Server Error</h1>
            <p>Something went wrong on our server.</p>
            <a href="/">Go to Homepage</a>
        </body>
        </html>
        ''', 500

@app.errorhandler(413)
def too_large(e):
    flash("File too large. Maximum file size is 16MB.", 'danger')
    return redirect(request.referrer or url_for('home'))

# ===================== MAIN ENTRY POINT =====================

if __name__ == '__main__':
    # Print startup information
    print("=" * 60)
    print("🚀 Health AI Assistant Starting...")
    print("=" * 60)
    print(f"📁 Flask Secret Key: {'✅ Set' if app.secret_key else '❌ Not Set'}")
   
    print(f"💾 Upload Folder: {app.config['UPLOAD_FOLDER']}")
    print(f"🌐 Server URL: http://localhost:5000")
    print("=" * 60)
    print("📋 Available Routes:")
    print("  /                     - Home page")
    print("  /chatbot_interface    - Full chatbot interface")
    print("  /api/chatbot-test     - Simple chatbot test")
    print("  /diabetes             - Diabetes prediction")
    print("  /heart                - Heart disease prediction")
    print("  /kidney               - Kidney disease prediction")
    print("  /api/status           - API status check")
    print("=" * 60)
    
    # Run the application
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )