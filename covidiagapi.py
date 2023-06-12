import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
model = None
is_deployed = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    # Perform checks for the availability of dependencies here

    # Check for xgboost
    try:
        import xgboost
        logger.info("xgboost is available.")
    except ImportError:
        logger.error("xgboost library is not available.")
        raise ImportError("xgboost library is not available.")

    # Check for numpy
    try:
        import numpy
        logger.info("numpy is available.")
    except ImportError:
        logger.error("numpy library is not available.")
        raise ImportError("numpy library is not available.")

    # Check for Flask
    try:
        import flask
        logger.info("Flask is available.")
    except ImportError:
        logger.error("Flask library is not available.")
        raise ImportError("Flask library is not available.")

    # Check for scikit-learn
    try:
        import sklearn
        logger.info("scikit-learn is available.")
    except ImportError:
        logger.error("scikit-learn library is not available.")
        raise ImportError("scikit-learn library is not available.")


def perform_initial_deployment_tasks():
    # Execute the deployment health check once during initial deployment
    deployment_health_check()


# Load model and dependencies
def load_model():
    global model, is_deployed
    if model is None:
        try:
            # Check the availability of dependencies
            check_dependencies()

            # Load the model
            model = xgb.XGBClassifier(random_state=30)
            model.load_model('model/covid_diag_model.json')
            logger.info("Model loaded successfully.")

            if not is_deployed:
                # Perform deployment-specific initialization
                perform_initial_deployment_tasks()
                is_deployed = True
        except Exception as ex:
            logger.error(f"Failed to load model: {str(ex)}")


# Custom deployment health check endpoint
@app.route('/deployment-health', methods=['GET'])
def deployment_health_check():
    try:
        # Check the availability of dependencies
        check_dependencies()

        # Return a success response indicating the deployment health
        logger.info("Deployment health check passed successfully.")
        return '', 200
    except Exception as ex:
        logger.error(f"Deployment health check failed: {str(ex)}")
        return 'Deployment health check failed', 500


# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    try:
        global model

        if model is None:
            # Model not loaded, load the model
            load_model()

        if model is not None:
            # Perform any health check logic here
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
    # Get data from POST request
    input_names = ['breath_input', 'fever_input', 'dry_input', 'sore_input', 'running_input', 'asthma_input',
                   'chronic_input', 'headache_input', 'heart_input', 'diabetes_input', 'hyper_input',
                   'fatigue_input', 'gastro_input', 'abroad_input', 'contact_input', 'attend_input',
                   'visit_input', 'family_input']

    input_values = {name: int(value) for name, value in request.form.items() if name in input_names and value.isdigit()}

    try:
        # Filter and convert to numpy array and reshape to one instance only
        integer_values = list(input_values.values())
        input_data = np.asarray(integer_values).reshape(1, -1)

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
