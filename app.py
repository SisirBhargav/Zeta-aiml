from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_placement():
    try:
        # Get form data
        cgpa = float(request.form.get('cgpa'))
        iq = int(request.form.get('iq'))
        profile_score = int(request.form.get('profile_score'))

        # Make prediction
        features = np.array([[cgpa, iq, profile_score]])  # Ensure correct shape
        prediction = model.predict(features)

        # Set result message
        if prediction[0] == 1:
            result_msg = 'Mallareddy Students will place'
        else:
            result_msg = 'Student needs to practice to get placed'

        return render_template('index.html', result=result_msg)
    
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)  # Debug mode enabled
