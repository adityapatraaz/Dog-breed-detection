from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Predictive function and model loading
class_labels = ['beagle', 'bulldog', 'dalmatian', 'german-shepherd', 'husky', 
                'labrador-retriever', 'poodle', 'rottweiler']

def predict_and_display(img):
    model = tf.keras.models.load_model('models/fine_tuned_inception.h5')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence_level = np.max(predictions) * 100
    predicted_class_name = class_labels[predicted_class]
    return predicted_class_name, confidence_level

@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_breed = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', filename)
            file.save(filepath)
            img = image.load_img(filepath, target_size=(224, 224))
            predicted_class, confidence = predict_and_display(img)
            predicted_breed = f"{predicted_class} (Confidence: {confidence:.2f}%)"
    return render_template('index.html', predicted_breed=predicted_breed)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
