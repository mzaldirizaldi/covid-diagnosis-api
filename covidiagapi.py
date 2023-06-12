from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np

app = Flask(__name__)
model = None

@app.before_request
def load_model():
    global model
    model = xgb.XGBClassifier(random_state=30)
    model.load_model('model/covid_diag_model.json')

@app.route('/', methods=['POST'])
def home():
    input_names = ['breath_input', 'fever_input', 'dry_input', 'sore_input', 'running_input', 'asthma_input',
                   'chronic_input', 'headache_input', 'heart_input', 'diabetes_input', 'hyper_input',
                   'fatigue_input', 'gastro_input', 'abroad_input', 'contact_input', 'attend_input',
                   'visit_input', 'family_input']

    input_values = {name: int(value) for name, value in request.form.items() if name in input_names and value.isdigit()}

    try:
        integer_values = list(input_values.values())
        input_data = np.asarray(integer_values).reshape(1, -1)

        pred_result_proba = model.predict_proba(input_data)
        pred_result = np.argmax(pred_result_proba)
        pred_result_proba = pred_result_proba[0, pred_result] * 100

        return jsonify({'pred_result_proba': str(pred_result_proba), 'pred_result': str(pred_result)})
    except (ValueError, KeyError):
        return jsonify({'error': 'Invalid input data'})

if __name__ == '__main__':
    app.run()