import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify
import logging
import importlib

app = Flask(__name__)
model = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    # Perform checks for the availability of dependencies here
    required_libraries = ['xgboost', 'numpy', 'flask', 'sklearn']
    missing_libraries = []

    for lib in required_libraries:
        try:
            importlib.import_module(lib)
            logger.info(f"{lib} is available.")
        except ImportError:
            logger.error(f"{lib} library is not available.")
            missing_libraries.append(lib)

    if missing_libraries:
        raise ImportError("One or more required libraries are not available.")


# Load model and check dependencies
def load_model():
    global model

    if model is None:
        try:
            # Check the availability of dependencies
            check_dependencies()

            # Load the model
            model = xgb.XGBClassifier(random_state=30)
            model.load_model('model/covid_diag_model.json')
            logger.info("Model loaded successfully.")
        except Exception as ex:
            logger.error(f"Failed to load model: {str(ex)}")
            return 'Model load failed', 500
    else:
        logger.info("Model is already loaded.")


# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    try:
        global model

        if model is None:
            # Model not loaded, load the model
            load_model()

        if model is not None:
            # Return a success response indicating the application is healthy
            logger.info("Health check passed successfully.")
            return '', 200
        else:
            # Model not loaded yet, treat it as a failure during initial deployment
            logger.error("Model not loaded.")
            return 'Health check failed', 500
    except Exception as ex:
        logger.error(f"Health check failed: {str(ex)}")
        return 'Health check failed', 500


@app.route('/', methods=['POST'])
def home():
    # Get data from POST request method
    input_names = [
        'breath_input', 'fever_input', 'dry_input', 'sore_input', 'running_input', 'asthma_input',
        'chronic_input', 'headache_input', 'heart_input', 'diabetes_input', 'hyper_input',
        'fatigue_input', 'gastro_input', 'abroad_input', 'contact_input', 'attend_input',
        'visit_input', 'family_input'
    ]

    input_values = {
        name: int(value) for name, value in request.form.items() if name in input_names and value.isdigit()
    }

    try:
        if model is None:
            logger.error("Model not loaded.")
            return jsonify({'error': 'Model not loaded'})

        # Filter and convert to numpy array and reshape to one instance only
        input_data = np.asarray(list(input_values.values())).reshape(1, -1)

        # Feed to model and predict
        pred_result_proba = model.predict_proba(input_data)
        pred_result = np.argmax(pred_result_proba)
        pred_result_proba = pred_result_proba[0, pred_result] * 100

        # Return result as json
        return jsonify({'pred_result_proba': str(pred_result_proba), 'pred_result': str(pred_result)})
    except (ValueError, KeyError) as ex:
        logger.error(f"Invalid input data: {str(ex)}")
        return jsonify({'error': 'Invalid input data'})


if __name__ == '__main__':
    try:
        logger.info("Starting the application.")
        app.run(host='0.0.0.0', port=8080, debug=False)
    except Exception as e:
        logger.error(f"An error occurred while running the application: {str(e)}")
