from flask import Flask, request, render_template
import joblib
import numpy as np
import sklearn

# Load the model and scaler
model = joblib.load('loan_classification_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')


# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        print(sklearn.__version__)


        # Collect user input
        age = float(request.form['age'])
        gender = int(request.form['gender'])
        experience = int(request.form['experience'])
        income = float(request.form['income'])
        education = int(request.form['education'])
        amount = float(request.form['amount'])

        # One-hot encode 'ownership'
        ownership = request.form['ownership']
        ownership_rent = 0
        ownership_own = 0
        ownership_other = 0
        if ownership == 'rent':
            ownership_rent = 1
        elif ownership == 'other':
            ownership_other = 1
        elif ownership == 'own':
            ownership_own = 1
            
        # Loan Intent
        intent = request.form['intent']
        intent_education = 0
        intent_medical = 0
        intent_venture = 0
        intent_personal = 0
        intent_home = 0
        if intent == 'education':
            intent_education = 1
        elif intent == 'home':
            intent_home = 1
        elif intent == 'personal':
            intent_personal = 1
        elif intent == 'venture':
            intent_venture = 1
        elif intent == 'medical':
            intent_medical = 1

        
        # Combine all features into a single NumPy array
        features = np.array([[age, gender, education, income, experience, amount, ownership_other, ownership_own, ownership_rent, intent_education, intent_home, intent_medical, intent_personal, intent_venture]])
        
        print(features)

        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Mapping prediction to species names
        status_mapping = {1: 'Approved', 0: 'Rejected'}
        predicted_loan_status = status_mapping.get(prediction, "Unknown")
        
        
        # Mapping prediction to species names
        # species_mapping = {1: 'Adelie', 2: 'Gentoo', 3: 'Chinstrap'}
        # predicted_species = species_mapping.get(prediction, "Unknown")
        
        return render_template('index.html', prediction_text=f'Predicted Loan Approval Status: {predicted_loan_status}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
