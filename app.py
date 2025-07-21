from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model('modelo_skin_cancer.h5')

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

@app.route('/predecir', methods=['POST'])
def predecir():
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400
    
    img = request.files['imagen']
    img_array = preprocess(img.read())
    
    pred = model.predict(img_array)
    resultado = "cáncer" if pred[0][0] > 0.5 else "no cáncer"
    
    return jsonify({'resultado': resultado})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
