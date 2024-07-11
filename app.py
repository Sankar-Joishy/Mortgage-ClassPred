import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from CustomPipeline import CustomPipeline




app = Flask(__name__)
model = pickle.load(open('pipeline.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [int(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)  # Reshape to match expected input shape

    # Ensure final_features has the same number of features as expected by the model
    if final_features.shape[1] != model.clf.n_features_in_:
        return render_template('index.html',
                               prediction_text='Not Eligible.')


    prediction = model.predict(final_features)

    output = round(prediction[0], 1)

    return render_template('index.html', prediction_text='Your Score is: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)