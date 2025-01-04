from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the best model
model = joblib.load('best_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    features = ['WorkoutTime', 'ReadingTime', 'PhoneTime', 'WorkHours', 'CaffeineIntake', 'RelaxationTime']
    input_data = []
    for feature in features:
        value = request.form.get(feature)
        try:
            input_data.append(float(value))
        except ValueError:
            return "Invalid input. Please enter numeric values."
    
    # Reshape for the model
    input_data = np.array(input_data).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Pass the prediction to result.html
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)