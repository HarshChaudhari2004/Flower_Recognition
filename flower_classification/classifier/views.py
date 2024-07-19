import os
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
import tensorflow as tf
from keras.models import load_model
import streamlit as st 
# import tensorflow.keras.models as models
import numpy as np

# Load the model once when the server starts
model_path = os.path.join(settings.BASE_DIR, 'classifier/models/Flower_Recog_Model.keras')
model = load_model(model_path)
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + flower_names[np.argmax(result)] + ' with a score of ' + str(np.max(result) * 100)
    return outcome

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        image_path = default_storage.save('uploads/' + image.name, image)
        image_full_path = os.path.join(settings.MEDIA_ROOT, image_path)
        classification_result = classify_images(image_full_path)
        context = {
            'image_url': default_storage.url(image_path),
            'classification_result': classification_result
        }
        return render(request, 'classifier/result.html', context)
    return render(request, 'classifier/upload.html')
