from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['POST'])
def home():
    # getting data from post
    input_names = ['breath_input', 'fever_input', 'dry_input', 'sore_input', 'running_input', 'asthma_input',
                   'chronic_input', 'headache_input', 'heart_input', 'diabetes_input', 'hyper_input',
                   'fatigue_input', 'gastro_input', 'abroad_input', 'contact_input', 'attend_input',
                   'visit_input', 'family_input']

    input_values = {name: int(value) for name, value in request.form.items() if name in input_names and value.isdigit()}
    integer_values = [value for value in input_values.values() if isinstance(value, int)]

    print(integer_values)

    # changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(integer_values)
    print(input_data_as_numpy_array)

    # reshaping the data for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    print(input_data_reshaped)

    # predicting
    pred_result_proba = xgb.predict_proba(input_data_reshaped)
    pred_result = np.argmax(pred_result_proba)
    pred_result_proba = pred_result_proba[0, pred_result] * 100
    print(pred_result_proba)
    print(pred_result)

    # return predicted values as json
    return jsonify({'pred_result_proba': str(pred_result_proba), 'pred_result': str(pred_result)})


if __name__ == '__main__':
    app.run(debug=True)
    xgb = xgb.XGBClassifier(random_state=30)
    xgb.load_model('covid_diag_model.json')
