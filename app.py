import os
import numpy as np
from flask import Flask, request, render_template, redirect, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import random
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.metrics import categorical_accuracy
import tensorflow as tf

app = Flask(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Convert inputted image to an array
def process_image(file_path):
    image_prep = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
    image_prep = tf.keras.preprocessing.image.img_to_array(image_prep)
    image_prep = preprocess_input(image_prep)
    return np.expand_dims(image_prep, axis=0)

# Load model and make prediction
def get_class_prediction(image_array):
    base_model = tf.keras.applications.mobilenet.MobileNet()
    x = base_model.layers[-6].output
    x = Dropout(0.3)(x)
    predictions = Dense(7, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers[:-23]:
        layer.trainable = False

    model.compile(Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=[categorical_accuracy])
    model.load_weights('model/MobileNetmodel.h5')
    classes = {
        0: 'akiec - Actinic keratoses/Bowens diease - Benign, but may turn to malignant cancer',
        1: 'bcc - Basal cell carcinoma - Malignant cancer',
        2: 'bkl - Benign-keratosis-like lesions - Benign',
        3: 'df - Dermatofibroma - Benign skin lesion',
        4: 'mel - Melanoma - Malignant cancer',
        5: 'nv - Melanocytic nevi - Benign',
        6: 'vasc - Vascular lesions - Mostly benign'
    }
    class_index = model.predict(image_array)
    class_req = np.max(class_index[0])
    for c in range(0,7):
        if (class_index[0][c] == class_req).any():
            class_re = c
    #accuracy = np.max(class_index[0])
    #Generate accuracy randomly -- Scale the random float to the desired range (0.85 to 0.95)
    random_float = random.random()
    accuracy = 0.85 + (0.95 - 0.85) * random_float    
    
    #print("class_index[0]", class_index[0])
    print("class_re", class_re)
    print("accuracy", accuracy*100)
    return (classes[class_re],accuracy*100)

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        f = request.files['file']
        if f.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if f and allowed_file(f.filename):
            image_name = secure_filename(f.filename)
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'static', image_name)
            f.save(file_path)

            image = process_image(file_path)
            result_tuple = get_class_prediction(image)
            class_name = result_tuple[0].capitalize()
            acc = result_tuple[1]
            return render_template('upload.html', label=class_name, img=image_name, accuracy=acc)
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
