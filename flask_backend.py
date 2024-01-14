from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('path/to/your/model')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json()

    # Perform predictions
    result = model.predict(data['input'])

    # Return the result as JSON
    return jsonify({'result': result.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
