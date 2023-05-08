import base64
import io
import pickle
import numpy as np
from PIL import Image
from flask import Flask, render_template, request , jsonify
import matplotlib.pyplot as plt

app = Flask(__name__,template_folder="templates")
print('star')
# Load the trained decision tree model
with open('best.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the symbol classes
symbol_classes = ['+', '-', 'x','÷', ')', '(']
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check')
def index2():
    print('check')
    return {"message":"Heyyyy"}, 200


@app.route('/predict', methods=['POST'])
def predict():

    base64_data = request.get_json()
    image_data = base64_data['image']
    padding = b'=' * (-len(image_data) % 4)
    image_data += padding.decode()

    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1]))).resize((28, 28)).convert('L')

    except:
        return {'error': 'Invalid image data'}

    image_array = np.array(image) / 255.0
    # print('image array = ',image_array.shape)
    # print(image_array.shape)
    prediction = model.predict(image_array.reshape(1, -1))[0]
    confidence = model.predict_proba(image_array.reshape(1, -1)).max()
    # print(prediction)
    # print('prediction = ', symbol_classes[prediction])
    # print('confidence = ', str(round(confidence*100, 2))+'%')
    result = {'prediction': symbol_classes[prediction], 'confidence': str(round(confidence*100, 2))+'%'}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
