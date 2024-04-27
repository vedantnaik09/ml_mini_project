from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the machine learning model
with open('ml_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    pelvic_incidence = float(request.form['pelvic_incidence'])
    pelvic_tilt = float(request.form['pelvic_tilt'])
    lumbar_lordosis_angle = float(request.form['lumbar_lordosis_angle'])
    sacral_slope = float(request.form['sacral_slope'])
    pelvic_radius = float(request.form['pelvic_radius'])
    degree_spondylolisthesis = float(request.form['degree_spondylolisthesis'])
    pelvic_slope = float(request.form['pelvic_slope'])
    direct_tilt = float(request.form['direct_tilt'])
    thoracic_slope = float(request.form['thoracic_slope'])
    cervical_tilt = float(request.form['cervical_tilt'])
    sacrum_angle = float(request.form['sacrum_angle'])
    scoliosis_slope = float(request.form['scoliosis_slope'])

    # Create a DataFrame with the user inputs
    user_inputs = pd.DataFrame({
        'pelvic_incidence': [pelvic_incidence],
        'pelvic_tilt': [pelvic_tilt],
        'lumbar_lordosis_angle': [lumbar_lordosis_angle],
        'sacral_slope': [sacral_slope],
        'pelvic_radius': [pelvic_radius],
        'degree_spondylolisthesis': [degree_spondylolisthesis],
        'pelvic_slope': [pelvic_slope],
        'direct_tilt': [direct_tilt],
        'thoracic_slope': [thoracic_slope],
        'cervical_tilt': [cervical_tilt],
        'sacrum_angle': [sacrum_angle],
        'scoliosis_slope': [scoliosis_slope]
    })

    # Make prediction using the loaded model
    prediction = model.predict(user_inputs)[0]

    return render_template('prediction.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

