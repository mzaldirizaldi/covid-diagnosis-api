import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
model = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    # Load model
    model = xgb.XGBClassifier(random_state=30)
    model.load_model('model/covid_diag_model.json')
    logging.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")


# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    # Perform any health check logic here
    # Return a success response indicating the application is healthy
    return jsonify({'status': 'ok'})


@app.route('/', methods=['POST'])
def home():
    # get data from POST
    input_names = ['breath_input', 'fever_input', 'dry_input', 'sore_input', 'running_input', 'asthma_input',
                   'chronic_input', 'headache_input', 'heart_input', 'diabetes_input', 'hyper_input',
                   'fatigue_input', 'gastro_input', 'abroad_input', 'contact_input', 'attend_input',
                   'visit_input', 'family_input']

    input_values = {name: int(value) for name, value in request.form.items() if name in input_names and value.isdigit()}

    try:
        # filter and convert to numpy array and reshape to one instance only
        integer_values = list(input_values.values())
        input_data = np.asarray(integer_values).reshape(1, -1)

        # feed to model and predict
        pred_result_proba = model.predict_proba(input_data)
        pred_result = np.argmax(pred_result_proba)
        pred_result_proba = pred_result_proba[0, pred_result] * 100

        # return result as json
        return jsonify({'pred_result_proba': str(pred_result_proba), 'pred_result': str(pred_result)})
    except (ValueError, KeyError):
        return jsonify({'error': 'Invalid input data'})


if __name__ == '__main__':
    try:
        logging.info("Starting the application.")
        app.run(host='0.0.0.0', port=8080, debug=False)
    except Exception as e:
        logger.error(f"An error occurred while running the application: {str(e)}")
